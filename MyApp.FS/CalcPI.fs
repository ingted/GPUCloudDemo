module MyApp.FS.CalcPI

open System
open Alea.CUDA
open Alea.CUDA.Unbound

// a gpu kernel, use [method-based way](http://quantalea.com/static/app/manual/compilation-method_based_gpu_coding.html)
[<ReflectedDefinition;AOTCompile(AOTOnly = true)>]
let kernelCountInside (pointsX:deviceptr<float>) (pointsY:deviceptr<float>) (numPoints:int) (numPointsInside:deviceptr<int>) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable i = start
    while i < numPoints do
        let x = pointsX.[i]
        let y = pointsY.[i]
        numPointsInside.[i] <- if sqrt (x*x + y*y) <= 1.0 then 1 else 0
        i <- i + stride

// in this demo, we use default gpu worker, so we check if it is available
let canDoGPUCalc =
    Lazy.Create <| fun _ ->
        try Device.Default |> ignore; true
        with _ -> false

// parameters of one calc task
type CalcParam =
    { Seed : uint32
      NumStreams : int
      NumPoints : int
      StreamId : int
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
            let streamId = param.StreamId

            let random = param.GetRandom seed numStreams numDimensions
            use reduce = DeviceSumModuleI32.Default.Create(numPoints)
            use points = random.AllocCUDAStreamBuffer(numPoints)
            use numPointsInside = worker.Malloc<int>(numPoints)
            let pointsX = points.Ptr
            let pointsY = points.Ptr + numPoints
            let lp = LaunchParam(numSMs * 8, 256)

            printfn "Random(%s) Streams(%d/%d) Points(%d)" (random.GetType().Namespace) (streamId+1) numStreams numPoints

            random.Fill(streamId, numPoints, points)
            worker.Launch <@ kernelCountInside @> lp pointsX pointsY numPoints numPointsInside.Ptr
            let numPointsInside = reduce.Reduce(numPointsInside.Ptr, numPoints)
            4.0 * (float numPointsInside) / (float numPoints)
            |> Some

    // if no gpu return none
    else None 

let test() =
    ()
//    let numPoints = 1000000
//    let numStreamsPerSM = 2
//    let getRandomXorshift7 numStreams numDimensions = Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, 42u) :> Rng.IRandom<float>
//    let getRandomMrg32k3a  numStreams numDimensions = Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, 42u) :> Rng.IRandom<float>
//
//    { TaskId = 0; NumPoints = numPoints; NumStreamsPerSM = numStreamsPerSM; GetRandom = getRandomXorshift7 }
//    |> calcPI |> printfn "pi=%A"
//
//    { TaskId = 0; NumPoints = numPoints; NumStreamsPerSM = numStreamsPerSM; GetRandom = getRandomMrg32k3a }
//    |> calcPI |> printfn "pi=%A"