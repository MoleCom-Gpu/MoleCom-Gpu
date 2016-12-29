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
-- It generates molecules for the transmitter. Algorithm works as follows
-- First generate coordinates of a molecule supposing the sphere centered at the origin, then move the points by center of the sphere
-- To generate coordinates first generate x and y coordinates then find z by the fact that a point on a surface of a sphere is exactly far from radius of that sphere.
function generateMolecules(transmitterNumber)
	-- radius of the given transmitter
	radius = transmittersRadius[transmitterNumber]
	-- randomly generate coordinates for x and y.
	moleculeX = torch.CudaTensor(1, symbolSize): uniform(-1 * radius, radius)
	moleculeY = torch.CudaTensor(1, symbolSize): uniform(-1 * radius, radius)
	-- if x^2 + y^2 >= r^2, set y^2 = r^2 - x^2 (This might be changed in order to improve randomness)
        sqMoleculeX = torch.cmul(moleculeX, moleculeX)
        sqMoleculeY = torch.cmul(moleculeY, moleculeY)
        flag = (sqMoleculeX + sqMoleculeY - radius * radius):ge(0) 
	subs = torch.pow(radius * radius - sqMoleculeX, 0.5)
        moleculeY = torch.cmul(flag, subs) + torch.cmul(1 - flag, moleculeY)
	-- now set z as sqrt(r^2-x^2-y^2)
         moleculeZ = torch.abs(radius * radius - sqMoleculeX - torch.cmul(moleculeY, moleculeY)) -- in order to avoid negative values caused by double precision.
	moleculeZ:pow(0.5)
        --print(moleculeX, moleculeY, moleculeZ)
	-- now randomly set sign of z coordinate (by square root only positive values are retrieved but the occurance of positive and negative values are equally likely. 
        signGenerator = torch.CudaTensor(1, symbolSize): uniform(-1, 1)
        signGenerator = signGenerator:ge(0)
	moleculeZ = torch.cmul(signGenerator, moleculeZ) + torch.cmul(1 - signGenerator, -moleculeZ)
	-- check: all values of this tensor should be radius^2 
	--print(torch.cmul(moleculeX, moleculeX) + torch.cmul(moleculeY, moleculeY) + torch.cmul(moleculeZ, moleculeZ))
	-- move all points by center of the sphere
	moleculeX:add(transmittersCoordinates[transmitterNumber][1])
	moleculeY:add(transmittersCoordinates[transmitterNumber][2])
	moleculeZ:add(transmittersCoordinates[transmitterNumber][3])
	return torch.cat(torch.cat(moleculeX, moleculeY, 1), moleculeZ, 1)
end
readConfiguration()
twoDT = (2 * diffusionCoefficient * deltaTime)
loopLength = runTime/deltaTime
symbolCheck = symbolDuration/deltaTime 
numberOfSymbols = runTime/symbolDuration
--numberOfMolecules = numberOfSymbols * symbolSize
molecules = generateMolecules(1)

availability = torch.CudaTensor(symbolSize): fill(1)
receiverCount = torch.CudaTensor(numberOfSymbols): fill(0) -- it might be also a double tensor
for i = 1, loopLength do
	if (i - 1) % symbolCheck == 0 and i > 1 then -- If there is time to generate new symbol and evaluate the previous one
		symbolNumber = (i-1) / symbolCheck 
		-- evaluation of previous symbol
		receiverCount[symbolNumber] = availability:size(1) - torch.sum(availability) - torch.sum(receiverCount)
		-- if symbol is 1, single transmitter
		molecules = torch.cat(molecules, generateMolecules(1))
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
