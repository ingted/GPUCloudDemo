#load "../packages/MBrace.Azure.Standalone/MBrace.Azure.fsx"

open MBrace.Azure

// place your connection strings here
let myStorageConnectionString = "yourstring"
let myServiceBusConnectionString = "yourstring"

let config =
    { Configuration.Default with
        StorageConnectionString = myStorageConnectionString
        ServiceBusConnectionString = myServiceBusConnectionString }
