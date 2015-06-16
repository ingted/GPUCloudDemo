module MyApp.FS.CalcPI

open System
open Alea.CUDA
open Alea.CUDA.Unbound

// GPU kernel to check if point is inside circle 
[<ReflectedDefinition;AOTCompile>]
let kernelCountInside (pointsX:deviceptr<float>) (pointsY:deviceptr<float>) (numPoints:int) (numPointsInside:deviceptr<int>) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable i = start
    while i < numPoints do
        let x = pointsX.[i]
        let y = pointsY.[i]
        numPointsInside.[i] <- if sqrt (x*x + y*y) <= 1.0 then 1 else 0
        i <- i + stride

// we use default gpu worker, so we check if it is available
let canDoGPUCalc =
    Lazy.Create <| fun _ ->
        try Device.Default |> ignore; true
        with _ -> false

// calculation task parameters 
type CalcParam =
    { Seed : uint32
      NumStreams : int
      NumPoints : int
      StartStreamId : int
      StopStreamId : int
      GetRandom : uint32 -> int -> int -> Rng.IRandom<float> }

let calcPI (param:CalcParam) =
    if canDoGPUCalc.Value then
        let worker = Worker.Default
        // switch to the gpu worker thread to execute GPU kernels
        worker.Eval <| fun _ ->
            let seed = param.Seed
            let numStreams = param.NumStreams
            let numDimensions = 2
            let numPoints = param.NumPoints
            let numSMs = worker.Device.Attributes.MULTIPROCESSOR_COUNT
            let startStreamId = param.StartStreamId
            let stopStreamId = param.StopStreamId

            let random = param.GetRandom seed numStreams numDimensions
            use reduce = DeviceSumModuleI32.Default.Create(numPoints)
            use points = random.AllocCUDAStreamBuffer(numPoints)
            use numPointsInside = worker.Malloc<int>(numPoints)
            let pointsX = points.Ptr
            let pointsY = points.Ptr + numPoints
            let lp = LaunchParam(numSMs * 8, 256)
            
            let pi =
                [| startStreamId .. stopStreamId |]
                |> Array.map (fun streamId ->
                    random.Fill(streamId, numPoints, points)
                    worker.Launch <@ kernelCountInside @> lp pointsX pointsY numPoints numPointsInside.Ptr
                    let numPointsInside = reduce.Reduce(numPointsInside.Ptr, numPoints)
                    4.0 * (float numPointsInside) / (float numPoints))
                |> Array.average

            printfn "Random(%s) Streams(%d-%d/%d) Points(%d) : %f" (random.GetType().Namespace) (startStreamId+1) (stopStreamId+1) numStreams numPoints pi

            Some pi

    // if no gpu return None
    else None 