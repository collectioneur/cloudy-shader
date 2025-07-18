import { useEffect, useMemo, useRef, useState } from "react";
import { Dimensions, GestureResponderEvent, View } from "react-native";
import { Canvas, useDevice, useGPUContext } from "react-native-wgpu";
import tgpu, { Render, Sampled, TgpuRoot, TgpuTexture } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { noiseTexture } from "./noiseTexture";

const MAX_ITERATIONS = 100;
const MAX_DIST = 100.0;
const SURFACE_DIST = 0.01;
const MARCH_SIZE = 0.05;

const mainVertex = tgpu["~unstable"].vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex },
  out: { outPos: d.builtin.position, uv: d.vec2f },
})/* wgsl */ `{
  var pos = array<vec2f, 6>(vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), 
    vec2(1.0, -1.0),
    vec2(1.0, 1.0));
  var uv = array<vec2f, 6>(
    vec2(0.0, 1.0), 
    vec2(0.0, 0.0),  
    vec2(1.0, 0.0),  
    vec2(0.0, 1.0),
    vec2(1.0, 0.0), 
    vec2(1.0, 1.0)   
  );
  return Out(vec4f(pos[in.vertexIndex], 0.0, 1.0), uv[in.vertexIndex]);
}`;

export default function Triangle() {
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  const { device = null } = useDevice();
  const root = useMemo(
    () => (device ? tgpu.initFromDevice({ device }) : null),
    [device]
  );
  const { ref, context } = useGPUContext();
  const time = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const pipelineRef = useRef<GPURenderPipeline | null>(null);
  const { width, height } = Dimensions.get("window");
  const w = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const h = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const finger = useMemo(() => root?.createUniform(d.vec2f) ?? null, [root]);
  const panFinger = useRef([-1, -1]);
  const panChange = useRef([0.0, 0.0]);
  const imageSampler = device?.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
    addressModeW: "repeat",
  });
  const [imageTexture, setImageTexture] = useState<
    (TgpuTexture & Sampled & Render) | undefined
  >(undefined);

  useEffect(() => {
    if (!root) {
      return;
    }

    async function init(root: TgpuRoot) {
      console.log("Initializing TypeGPU...");
      const response = await fetch(noiseTexture);
      const imageBitmap = await createImageBitmap(await response.blob());
      const [srcWidth, srcHeight] = [imageBitmap.width, imageBitmap.height];

      const image = root["~unstable"]
        .createTexture({
          size: [srcWidth, srcHeight],
          format: "rgba8unorm",
        })
        .$usage("sampled", "render");

      setImageTexture(image);

      root.device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: root.unwrap(image) },
        [srcWidth, srcHeight]
      );
    }

    init(root);
  }, [root]);

  const handleTouchMove = (e: GestureResponderEvent) => {
    const touch = e.nativeEvent.touches[0];
    const { locationX, locationY, pageX, pageY } = touch;
    if (panFinger.current[0] !== -1) {
      const dx = locationX - panFinger.current[0];
      const dy = locationY - panFinger.current[1];
      panChange.current = [
        panChange.current[0] + dx,
        Math.min(Math.max(-height * 3, panChange.current[1] + dy), height * 3),
      ];
      console.log("Pan change:", -panChange.current[1] / height);
      finger?.write(
        d.vec2f(panChange.current[0] / width, -panChange.current[1] / height)
      );
    }
    panFinger.current = [locationX, locationY];
  };

  const handleEndTouch = () => {
    console.log("Touch ended");
    panFinger.current = [-1, -1];
  };

  useEffect(() => {
    if (!root || !device || !context) return;
    if (!imageTexture) return;
    if (w === null || h === null || time === null) return;
    w.write(width);
    h.write(height);

    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "premultiplied",
    });

    const sampledView = imageTexture.createView("sampled");
    const sampler = tgpu["~unstable"].sampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    const getNormal = tgpu.fn(
      [d.vec3f],
      d.vec3f
    )((p) => {
      let e = d.vec2f(0.01, 0.0);
      let xyy = d.vec3f(e.x, e.y, e.y);
      let yxy = d.vec3f(e.y, e.x, e.y);
      let yyx = d.vec3f(e.y, e.y, e.x);

      let n = std.sub(
        scene(p),
        d.vec3f(
          scene(std.sub(p, xyy)),
          scene(std.sub(p, yxy)),
          scene(std.sub(p, yyx))
        )
      );
      return std.normalize(n);
    });

    const noise = tgpu.fn(
      [d.vec3f],
      d.f32
    )((x) => {
      let p = std.floor(x);
      let f = std.fract(x);

      f = std.mul(std.mul(f, f), std.sub(3.0, std.mul(2.0, f)));

      let uv = std.add(
        std.add(p.xy, std.mul(d.vec2f(37.0, 239.0), d.vec2f(p.z, p.z))),
        f.xy
      );
      let tex = std.textureSampleLevel(
        sampledView,
        sampler,
        std.fract(std.div(std.add(uv, d.vec2f(0.5, 0.5)), 256.0)),
        0.0
      ).yx;

      return std.mix(tex.x, tex.y, f.z) * 2.0 - 1.0;
      // return u.x;
    });

    const fbm = tgpu.fn(
      [d.vec3f],
      d.f32
    )((p) => {
      let q = std.add(
        p,
        d.vec3f(std.sin(time.$), std.cos(time.$), time.$ * 3.0)
      );
      let f = d.f32(0.0);
      let scale = d.f32(1.0);
      let factor = d.f32(1.8);

      for (let i = 0; i < 4; i++) {
        f += noise(q) * scale;
        q = std.mul(q, factor);
        scale *= 0.4;
        factor += 0.5;
      }
      return f;
    });

    const sdSphere = tgpu.fn(
      [d.vec3f, d.f32],
      d.f32
    )((p, r) => {
      return std.length(p) - r;
    });

    const scene = tgpu.fn(
      [d.vec3f],
      d.f32
    )((p) => {
      let distance = sdSphere(p, 2.0);

      let f = fbm(p);
      return f - 0.5;
    });

    const raymarch = tgpu.fn(
      [d.vec3f, d.vec3f],
      d.vec4f
    )((ro, rd) => {
      let res = d.vec4f(0.0, 0.0, 0.0, 0.0);
      let transparency = 0.0;
      let hash = std.fract(
        std.sin(std.dot(rd.xy, d.vec2f(12.9898, 78.233))) * 43758.5453
      );
      let depth = hash * MARCH_SIZE;
      for (let i = 0; i < MAX_ITERATIONS; i++) {
        let p = std.add(ro, std.mul(rd, depth));
        let density = std.clamp(scene(p), 0.0, 1.0);
        if (density > 0.0) {
          let sunDirection = std.normalize(d.vec3f(1.0, 0.0, 0.0));
          let diffuse = std.clamp(
            (scene(p) - scene(std.add(p, std.mul(0.8, sunDirection)))) / 0.8,
            0.0,
            1.0
          );
          diffuse = std.mix(0.5, 1.0, diffuse);
          let lin = std.add(
            std.mul(d.vec3f(0.6, 0.5, 0.75), 1.1),
            std.mul(d.vec3f(1.0, 0.7, 0.3), diffuse * 0.8)
          );
          let color = d.vec4f(
            std.mix(d.vec3f(1.0, 1.0, 1.0), d.vec3f(0.0, 0.0, 0.0), density),
            density
          );
          color = d.vec4f(
            color.x * lin.x,
            color.y * lin.y,
            color.z * lin.z,
            color.w
          );
          color = d.vec4f(
            color.x * color.w,
            color.y * color.w,
            color.z * color.w,
            color.w
          );
          res = std.add(res, std.mul(color, 0.9 - res.w));
        }
        depth += MARCH_SIZE;
      }
      return res;
    });

    const mainFragment = tgpu["~unstable"].fragmentFn({
      in: { uv: d.vec2f },
      out: d.vec4f,
    })(({ uv }) => {
      {
        let lightPos = d.vec3f(2.0, 0.0, -3.0);
        let new_uv = (uv - 0.5) * 2.0;
        new_uv.y *= h.$ / w.$;
        // let ro = d.vec3f(
        //   std.cos(time.$),
        //   std.cos(time.$ * 4),
        //   -std.abs(std.sin(time.$ * 4)) * 5.0 - 1.0
        // );
        let ro = d.vec3f(0.0, 0.0, -3.0);
        let rd = std.normalize(d.vec3f(new_uv.xy, 1.0));

        let color = d.vec3f(0.7, 0.7, 0.9);
        color -= 0.3 * d.vec3f(1, 0.85, 0.48) * rd.y;
        // color += 0.5 * d.vec3f(1.0,0.5,0.3) * pow(sun, 10.0);

        let res = raymarch(ro, rd);
        color = color * (1.1 - res.a) + res.rgb;
        // color = std.mix(color, res.xyz, res.w);

        return d.vec4f(color, 1.0);
      }
    });

    // const mainFragment = tgpu["~unstable"].fragmentFn({
    //   in: { uv: d.vec2f },
    //   out: d.vec4f,
    // })/* wgsl */ `{
    //   let color = textureSample(sampledView, sampler, in.uv).rgb;
    //   return vec4f(color, 1.0);
    // }`.$uses({ sampledView, sampler });

    const pipeline = root["~unstable"]
      .withVertex(mainVertex, {})
      .withFragment(mainFragment, { format: presentationFormat })
      .createPipeline();

    const startTime = performance.now();
    let frameId: number;

    const render = () => {
      const timestamp = (performance.now() - startTime) / 1000;
      time.write(timestamp);

      const view = context.getCurrentTexture().createView();

      pipeline
        .withColorAttachment({
          view,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        })
        .draw(6);

      context.present();
      frameId = requestAnimationFrame(render);
    };

    frameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(frameId);
      // root.destroy();
    };
  }, [device, context, imageTexture]);

  return (
    <View
      style={{ flex: 1, backgroundColor: "red" }}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleEndTouch}
    >
      <Canvas ref={ref} style={{ flex: 1, backgroundColor: "black" }} />
    </View>
  );
}
