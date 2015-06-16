#load "Config.fsx"

open MBrace.Azure.Client

// use this script to start a local worker on your machine
let cluster = Runtime.GetHandle(Config.config)
cluster.AttachLocalWorker(1, 1)

// you can optionally clear all processes
cluster.ClearAllProcesses()
