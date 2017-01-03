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
numberOfTransmitters = 0

moleculeRadius = 0
receiversCoordinates = {}
transmittersCoordinates = {}
receiversRadius = {}
transmittersRadius = {}
-- Reads the configuration file
function readConfiguration()

	print 'Reading configuration file...'
	mode = 0
	file = torch.DiskFile('config', 'r')
	configurationObject = file:readObject()

	diffusionCoefficient = configurationObject.diffusionCoefficient
	deltaTime = configurationObject.deltaTime
	runTime = configurationObject.runTime
	symbolSize = configurationObject.symbolSize
	symbolDuration = configurationObject.symbolDuration
	numberOfReceivers = configurationObject.numberOfReceivers
	numberOfTransmitters = configurationObject.numberOfTranmitters

	receiversCoordinates = torch.CudaTensor(numberOfReceivers, 3)
	transmittersCoordinates = torch.CudaTensor(numberOfTransmitters, 3)
	receiversRadius = torch.CudaTensor(numberOfReceivers)
	transmittersRadius = torch.CudaTensor(numberOfTransmitters)
	
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
-- TODO: kill the program if the configuration is broken, i.e wrong number of transmitters, etc..


-- Set some variables related to simulation
twoDT = torch.pow(2 * diffusionCoefficient * deltaTime, 0.5)
loopLength = runTime/deltaTime
symbolCheck = symbolDuration/deltaTime 
numberOfSymbols = runTime/symbolDuration
--numberOfMolecules = numberOfSymbols * symbolSize
-- This tensor holds the number of received molecules in each symbol duration for each receiver
receiverCount = torch.CudaTensor(numberOfReceivers, numberOfSymbols): fill(0) -- it might be also a double tensor
-- generate first iteration of molecules
-- to do concat. first initialize molecules
molecules = generateMolecules(1)
availability = torch.CudaTensor(symbolSize): fill(1)
-- then do for other transmitters.
for i=2, numberOfTransmitters do
	molecules = torch.cat(molecules, generateMolecules(i))
	availability = torch.cat(availability, torch.CudaTensor(symbolSize): fill(1))
end
for i = 1, loopLength do
	if (i - 1) % symbolCheck == 0 and i > 1 then -- If there is time to generate new symbol and evaluate the previous one
		symbolNumber = (i-1) / symbolCheck 
		-- evaluation of previous symbol
		receiverCount[1][symbolNumber] = availability:size(1) - torch.sum(availability) - torch.sum(receiverCount)
		for i=1, numberOfTransmitters do
        		molecules = torch.cat(molecules, generateMolecules(i))
        		availability = torch.cat(availability, torch.CudaTensor(symbolSize): fill(1))
     		end
		
		
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
receiverCount[1][numberOfSymbols] = availability:size(1) - torch.sum(availability) - torch.sum(receiverCount)
file = io.open('results.txt', 'w')
file:write('Simulation results\n')
file:write('Total number of received molecules\n')
file:write(availability:size(1) - torch.sum(availability))
file:write('\nTotal number of released molecules\n')
file:write(availability:size(1))
file:write("\n\n")
file:write("Received molecules per symbol duration for Receiver 1\n")
for i=1, receiverCount:size(2) do
	file:write(i)
	file:write(" => ")
	file:write(receiverCount[1][i])
	file:write("\n")
end


--print(receiverCount[1])

