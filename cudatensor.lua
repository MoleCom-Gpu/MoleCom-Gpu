--USE EVERY UNIT WITH MICROMETER!

cutorch = require 'cutorch'

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
	diffusionCoefficient = 79.4
	deltaTime = 0.005
	runTime = 100
	symbolSize = 1000
	symbolDuration = 1
	numberOfReceivers = 1
	numberOfTranmitters = 1
	receiversCoordinates = torch.CudaTensor(numberOfReceivers, 3)
	transmittersCoordinates = torch.CudaTensor(numberOfTranmitters, 3)
	receiversRadius = torch.CudaTensor(numberOfReceivers)
	transmittersRadius = torch.CudaTensor(numberOfTranmitters)
	
	receiversCoordinates[1] = torch.CudaTensor({60, 60, 60})
	transmittersCoordinates[1] = torch.CudaTensor({40, 40, 40})
	receiversRadius[1] = 10
	transmittersRadius[1] = 10
	moleculeRadius = 2.5e-3
	
end

readConfiguration()
twoDT = (2 * diffusionCoefficient * deltaTime)
loopLength = runTime/deltaTime
symbolCheck = symbolDuration/deltaTime 
numberOfSymbols = runTime/symbolDuration
numberOfMolecules = numberOfSymbols * symbolSize
molecules = torch.CudaTensor(3, numberOfMolecules): uniform(0, 100)
availability = torch.CudaTensor(numberOfMolecules): fill(0)
for i = 1, loopLength do
	if (i - 1) % symbolCheck == 0 then
		symbolNumber = (i-1) / symbolCheck
		for j = symbolNumber * symbolSize + 1, (symbolNumber + 1) * symbolSize do
			availability[j] = 1
		end  
	end
	--if i < 10 then
	--	print(i, molecules[1][1], molecules[2][1], molecules[3][1])
	--end
	delta1 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT)
	delta2 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT)  
	delta3 = torch.CudaTensor(numberOfMolecules): normal(0, twoDT) 
	molecules[1]: add(delta1:cmul(availability))
	molecules[2]: add(delta2:cmul(availability))
	molecules[3]: add(delta3:cmul(availability))
	--if i < 10 then
	--	print(i, molecules[1][1], molecules[2][1], molecules[3][1])
	--end
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
	--print(dist)
	availability = dist:gt(receiversRadius[1])
	--if i < 30 then
	--	print(i, availability[1], dist[1])
	--end	
end
--print(availability)
y = torch.sum(availability)
--print(availability)
print(100 * (numberOfMolecules - y) / numberOfMolecules)
print(numberOfMolecules)
--print(availability)
