import { useEffect, useMemo, useRef, useState } from "react";
import { Dimensions } from "react-native";
import { Canvas, useDevice, useGPUContext } from "react-native-wgpu";
import tgpu, {
  Render,
  Sampled,
  TgpuRoot,
  TgpuSampledTexture,
  TgpuTexture,
} from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { noiseTexture } from "./noiseTexture";

const MAX_ITERATIONS = 120; // 50 - 200
const MARCH_SIZE = 0.05; // 0.05 - 0.15
const SUN_DIRECTION = d.vec3f(1.0, 0.0, 0.0); // [-1.0, -1.0, -1.0] - [1.0, 1.0, 1.0]
const ANGLE_DISTORTION = 1.0; // 0.1 - 3.0
const SUN_INTENSITY = 0.7; // 0.01 - 1.0
const LIGHT_ABSORBTION = 0.88; // 0.0 - 1.0
const CLOUD_DENSITY = 0.6; // 0.0 - 1.0
const CLOUD_CORE_DENSITY = 1.0; //0.0 - 10.0
const FLIGHT_SPEED = 3.0; // 1.0 - 10.0
const CLOUD_DETALIZATION = 2.23; // 0.0 - 4.0

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

export default function Clouds() {
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
  const imageSampler = device?.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });
  const [imageTexture, setImageTexture] = useState<
    (TgpuTexture & Sampled & Render) | undefined
  >(undefined);

  useEffect(() => {
    if (!root) {
      return;
    }

    async function init(root: TgpuRoot) {
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

    const sampledView = imageTexture.createView(
      "sampled"
    ) as TgpuSampledTexture<"2d", d.F32>;
    const sampler = tgpu["~unstable"].sampler({
      magFilter: "linear",
      minFilter: "linear",
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
    });

    const fbm = tgpu.fn(
      [d.vec3f],
      d.f32
    )((p) => {
      let q = std.add(
        p,
        d.vec3f(std.sin(time.$), std.cos(time.$), time.$ * FLIGHT_SPEED)
      );
      let f = d.f32(0.0);
      let scale = d.f32(CLOUD_CORE_DENSITY);
      let factor = d.f32(CLOUD_DETALIZATION);

      for (let i = 0; i < 4; i++) {
        f += noise(q) * scale;
        q = std.mul(q, factor);
        scale *= 0.4;
        factor += 0.5;
      }
      return f;
    });

    const scene = tgpu.fn(
      [d.vec3f],
      d.f32
    )((p) => {
      let f = fbm(p);
      return f - 1.5 + CLOUD_DENSITY * 2.0;
    });

    const raymarch = tgpu.fn(
      [d.vec3f, d.vec3f, d.vec3f],
      d.vec4f
    )((ro, rd, sunDirection) => {
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
          let diffuse = std.clamp(
            scene(p) - scene(std.add(p, sunDirection)),
            0.0,
            1.0
          );
          diffuse = std.mix(0.3, 1.0, diffuse);
          let lin = std.add(
            std.mul(d.vec3f(0.6, 0.45, 0.75), 1.1),
            std.mul(d.vec3f(1.0, 0.7, 0.3), diffuse * SUN_INTENSITY)
          );
          let color = d.vec4f(
            std.mix(d.vec3f(1.0, 1.0, 1.0), d.vec3f(0.2, 0.2, 0.2), density),
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
          res = std.add(res, std.mul(color, LIGHT_ABSORBTION - res.w));
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
        let new_uv = std.mul(std.sub(uv, 0.5), 2.0);
        new_uv = d.vec2f(new_uv.x, new_uv.y * (h.$ / w.$));
        let sunDirection = std.normalize(SUN_DIRECTION);
        let ro = d.vec3f(0.0, 0.0, -3.0);
        let rd = std.normalize(d.vec3f(new_uv.x, new_uv.y, ANGLE_DISTORTION));
        let sun = std.clamp(std.dot(rd, sunDirection), 0.0, 1.0);

        let color = d.vec3f(0.75, 0.66, 0.9);

        color = std.sub(color, std.mul(0.35 * rd.y, d.vec3f(1, 0.7, 0.43)));

        color = std.add(
          color,
          std.mul(
            d.vec3f(1.0, 0.37, 0.17),
            std.pow(sun, 1.0 / std.pow(SUN_INTENSITY, 3.0))
          )
        );

        let res = raymarch(ro, rd, sunDirection);

        color = std.add(std.mul(color, 1.1 - res.w), res.xyz);

        return d.vec4f(color, 1.0);
      }
    });

    const pipeline = root["~unstable"]
      .withVertex(mainVertex, {})
      .withFragment(mainFragment, { format: presentationFormat })
      .createPipeline();

    let startTime = performance.now();
    let frameId: number;

    const render = () => {
      const timestamp = (performance.now() - startTime) / 1000;
      if (timestamp > 500.0) {
        startTime = performance.now();
      }
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

  return <Canvas ref={ref} style={{ flex: 1 }} />;
}
