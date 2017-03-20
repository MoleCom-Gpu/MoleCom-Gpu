cutorch = require 'cutorch'

transmittersRadius = {}
receiversCoordinates = {}
tranmittersCoordinates = {}
receiversRadius = {}
tranmittersRadius = {}


print 'Generating configuration file'
mode = 0
file = torch.DiskFile('config', 'w')
	--configurationString = file:readString("*a")

local configurationFile = {
	diffusionCoefficient = 79.4 ,
	deltaTime = 0.005 ,
	runTime = 10 ,
	symbolSize = 100 ,
	symbolDuration = 1 ,
	numberOfReceivers = 1 ,
	numberOfTranmitters = 1 ,

	receiversCoordinates = { torch.DoubleTensor({60, 60, 60}) },
	transmittersCoordinates = { torch.DoubleTensor({40, 40, 40}) },
	receiversRadius = {10},
	transmittersRadius ={10} ,
	moleculeRadius = 2.5e-3
}

file:writeObject(configurationFile)
file:close()

file = torch.DiskFile('config', 'r')
configurationObject = file:readObject()
print (configurationObject)

print 'Generation completed'
