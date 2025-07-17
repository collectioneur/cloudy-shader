import { useEffect, useMemo, useRef } from "react";
import { Dimensions } from "react-native";
import { Canvas, useDevice, useGPUContext } from "react-native-wgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

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

const smoothstep = tgpu.fn([d.f32, d.f32, d.f32], d.f32)`(edge0, edge1, x) {
  if (edge0 == edge1) {
    return select(0.0, 1.0, x >= edge1);
  }
  let t = clamp(
    (x - edge0) / (edge1 - edge0),
    0.0,
    1.0
  );
  return t * t * (3.0 - 2.0 * t);
}`;

const step = tgpu.fn([d.f32, d.f32], d.f32)`(edge, x) {
  if (x < edge) {
    return 0.0;
  } else {
    return 1.0;
  }
}`;

const palette = tgpu.fn([d.f32], d.vec3f)`(t) {
  // let a = vec3f(0.50,0.59,0.85);
  // let b = vec3f(0.18,0.42,0.40);
  // let c = vec3f(0.18,0.48,0.41);
  // let d = vec3f(0.35,0.13,0.32);

  let a = vec3f(0.71,0.08,0.69);
  let b = vec3f(0.50,0.05,0.31);
  let c = vec3f(0.69,0.08,0.89);
  let d = vec3f(0.81,0.63,0.58);

  return a + b * cos(6.28318 * (c * t + d));
}`;

const mod = tgpu.fn(
  [d.vec2f, d.f32],
  d.vec2f
)`(v, a) { return fract(v / a) * a;}`;

const length = tgpu.fn(
  [d.vec2f],
  d.f32
)`(v) { return sqrt(v.x * v.x + v.y * v.y); }`;

export default function Triangle() {
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  const { device = null } = useDevice();
  const root = useMemo(
    () => (device ? tgpu.initFromDevice({ device }) : null),
    [device]
  );
  const { ref, context } = useGPUContext();
  const time = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);

  const pipelineRef = useRef<GPUComputePipeline | null>(null);
  const { width, height } = Dimensions.get("window");
  const w = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);
  const h = useMemo(() => root?.createUniform(d.f32) ?? null, [root]);

  useEffect(() => {
    if (!root) return;
    pipelineRef.current = root["~unstable"]
      .withVertex(mainVertex, {})
      .withFragment(mainFragment, { format: presentationFormat })
      .createPipeline();
    return () => root.destroy();
  }, [root, presentationFormat]);

  const mainFragment = tgpu["~unstable"].fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    {
      const newuv: d.vec2f = (uv.xy - 0.5) * 2;
      newuv.y *= h.$ / w.$;
      let uvv = newuv;
      let finalColor = d.vec3f(0.0, 0.0, 0.0);
      for (let i = 0.0; i < 3.0; i++) {
        newuv = std.fract(newuv * -0.9) - 0.5;
        let len = length(newuv) * std.exp(-length(uvv) * 0.5);
        let col = palette(length(uvv) + time.$ * 0.9);
        len = std.sin(len * 8 + time.$) / 8;
        len = std.abs(len);
        len = smoothstep(0.0, 0.1, len);
        len = 0.1 / len;
        finalColor.x += col.x * len;
        finalColor.y += col.y * len;
        finalColor.z += col.z * len;
      }
      return d.vec4f(finalColor, 1.0);
    }
  });

  useEffect(() => {
    if (!root || !device || !context) return;
    if (w === null || h === null || time === null) return;
    w.write(width);
    h.write(height);

    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "premultiplied",
    });

    const startTime = performance.now();
    let frameId: number;

    const render = () => {
      const timestamp = (performance.now() - startTime) / 1000;
      time.write(timestamp);

      const view = context.getCurrentTexture().createView();

      pipelineRef.current
        .withColorAttachment({
          view,
          clearValue: [0, 0, 0, 0],
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
      root.destroy();
    };
  }, [device, context]);

  return (
    <>
      <Canvas  ref={ref} style={{ flex: 1, backgroundColor: "black" }} />
    </>
  );
}
