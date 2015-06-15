#load "Common.fsx"

open MBrace.Azure.Client
open CloudScripts

let cluster = Runtime.GetHandle(config)

cluster.ShowWorkers()
cluster.ShowProcesses()

cluster.AttachLocalWorker(1, 1)
cluster.ClearAllProcesses()
