--USE EVERY UNIT WITH MICROMETER!

cutorch = require 'cutorch'
local  LIP = require 'JSON'

diffusionCoefficient = 0
deltaTime = 0
runTime = 0
symbolDuration = 0
--for now, assume no boundary.

symbolSize = 0
numberOfReceivers = 0
numberOfTranmitters = 0

moleculeRadius = 0
receiversCoordinates = {}
tranmittersCoordinates = {}
receiversRadius = {}
tranmittersRadius = {}

-- Reads the configuration file
function readConfiguration()

	print 'Reading configuration file...'
	mode = 0
	file = torch.DiskFile('config2', 'r')
	configurationObject = file:readObject()



	diffusionCoefficient = configurationObject.diffusionCoefficient
	deltaTime = configurationObject.deltaTime
	runTime = configurationObject.runTime
	symbolSize = configurationObject.symbolSize
	symbolDuration = configurationObject.symbolDuration
	numberOfReceivers = configurationObject.numberOfReceivers
	numberOfTranmitters = configurationObject.numberOfTranmitters

	receiversCoordinates = torch.CudaTensor(numberOfReceivers, 3)
	transmittersCoordinates = torch.CudaTensor(numberOfTranmitters, 3)
	receiversRadius = torch.CudaTensor(numberOfReceivers)
	transmittersRadius = torch.CudaTensor(numberOfTranmitters)
	
	for i,v in ipairs(configurationObject.receiversCoordinates) do
		receiversCoordinates[i] = v;
	end

	for i,v in ipairs(configurationObject.transmittersCoordinates) do
		transmittersCoordinates[i] = v;
	end

	for i,v in ipairs(configurationObject.receiversRadius) do
		receiversRadius[i] = v;
	end

	for i,v in ipairs(configurationObject.transmittersRadius) do
		transmittersRadius[i] = v;
	end

	moleculeRadius = configurationObject.moleculeRadius
	print 'Configuration file is read.'
	
end

readConfiguration()
twoDT = (2 * diffusionCoefficient * deltaTime)
loopLength = runTime/deltaTime
symbolCheck = symbolDuration/deltaTime 
numberOfSymbols = runTime/symbolDuration
--numberOfMolecules = numberOfSymbols * symbolSize
molecules = torch.CudaTensor(3, symbolSize): uniform(30, 50)
availability = torch.CudaTensor(symbolSize): fill(1)
receiverCount = torch.CudaTensor(numberOfSymbols): fill(0) -- it might be also a double tensor
for i = 1, loopLength do
	if (i - 1) % symbolCheck == 0 and i > 1 then -- If there is time to generate new symbol and evaluate the previous one
		symbolNumber = (i-1) / symbolCheck 
		-- evaluation of previous symbol
		receiverCount[symbolNumber] = availability:size(1) - torch.sum(availability) - torch.sum(receiverCount)
		-- if symbol is 1, single transmitter
		molecules = torch.cat(molecules, torch.CudaTensor(3, symbolSize): uniform(0, 100))
		availability = torch.cat(availability, torch.CudaTensor(symbolSize):fill(1))
		
	end
	-- Move all available molecules. Received ones are not moved since deltas are multiplying with availability flag.
	numberOfMolecules = molecules:size(2)
	delta1 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT)
	delta2 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT)  
	delta3 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT) 
	molecules[1]: add(delta1:cmul(availability))
	molecules[2]: add(delta2:cmul(availability))
	molecules[3]: add(delta3:cmul(availability))

	-- single receiver.
	dd1 = molecules[1] - receiversCoordinates[1][1]
	dd2 = molecules[2] - receiversCoordinates[1][2]
	dd3 = molecules[3] - receiversCoordinates[1][3]
	sq1 = torch.cmul(dd1, dd1)
	sq2 = torch.cmul(dd2, dd2)
	sq3 = torch.cmul(dd3, dd3)
	dist = sq1:add(sq2)
	dist:add(sq3)
	dist:pow(0.5)
	availability = dist:gt(receiversRadius[1])
end
--last symbol
receiverCount[numberOfSymbols] = availability:size(1) - torch.sum(availability) - torch.sum(receiverCount)
print('Simulation is finished')
print(receiverCount)
print('Total number of received molecules')
print(availability:size(1) - torch.sum(availability))
print('Total number of released molecules')
print(availability:size(1))
