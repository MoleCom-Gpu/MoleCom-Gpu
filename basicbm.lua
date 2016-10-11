-- Kerim Gokarslan <kerim.gokarslan@boun.edu.tr> 2016.
-- Basic implementation of Brownian Motion using CUDA.

-- Usage:
--

--PACKAGES (To make the runs short, one can require these packages in the Torch terminal for once and comment out these lines.
--cutorch = require 'cutorch'
--nngraph = require 'nngraph'

-- This code sets all tensors as CudaTensor but it is not a good approach since some of torch functionalities(eg: random) is not implemented according to their documentation
--torch.setdefaulttensortype('torch.CudaTensor')
-- Indexes of items those are read from the configuration file.
indexTransmitter = 1
indexReceiver = 2
indexMolecule = 3
indexDimension = 4
indexRunTime = 5
indexDeltaTime = 6
-- Holds the number of items read from the configuration file
configData = torch.CudaTensor(6)

-- Configuration file reader.
function readConfiguration()
	print 'Reading configuration file...'
	mode = 0
	file = torch.DiskFile('config', 'r')
	configurationString = file:readString("*a")
	local x, a, b = 1
	while x<string.len(configurationString) do
		a, b = string.find(configurationString, '.-\n', x)
		if not a then
			print 'unexpected eof, did you forget to add end-of-line?'
			break;
		else
			line = string.sub(configurationString, a, b)
			if string.find(line, "#") then
				--print(line)
				mode = mode + 1
			--elseif string.find(line, "\n") then
			
			else 
				configData[mode] = tonumber(line)
			end
		end
		x = b + 1
	end
	print 'Configuration file is read.'
end
function euclidean(point1, point2)
	d1 = (point1[1] - point2[1]) * (point1[1] - point2[1]);
	d2 = (point1[2] - point2[2]) * (point1[2] - point2[2]);
	d3 = (point1[3] - point2[3]) * (point1[3] - point2[3]);
	return torch.sqrt(d1 + d2 + d3)
end
function check() -- the probability of two molecules in the same coordinates are ignored.
	for i = 1, configData[indexMolecule] do
		if moleculeAvailability[i] == 1 then
			if molecules[i][1] < 0 then
				molecules[i][1] = 0
			end
			if molecules[i][2] < 0 then
				molecules[i][2] = 0
			end
			if molecules[i][2] < 0 then
				molecules[i][2] = 0
			end
			for j=1, configData[indexReceiver] do
				if euclidean(molecules[i], receivers[j]) <= 1 then
					print(i .. 'th molecule is received.')
					moleculeAvailability[i] = 0
					molecules[i] = torch.CudaTensor{-1, -1, -1}
					numberOfMoleculesReceived = numberOfMoleculesReceived + 1
				elseif euclidean(molecules[i], transmitters[j]) <=1 then
					print(i .. 'th molecule bounced from a transmitter.')
					vX = molecules[i][1] - transmitters[j][1]
					vY = molecules[i][2] - transmitters[j][2]
					vZ = molecules[i][3] - transmitters[j][3]
					magV = torch.sqrt(vX*vX + vY*vY + vZ*vZ);
					molecules[i][1] = transmitters[j][1] + vX / magV * 1; -- 1 is radius
					molecules[i][2] = transmitters[j][2] + vY / magV * 1; -- 1 is radius
					molecules[i][3] = transmitters[j][3] + vZ / magV * 1; -- 1 is radius
					--print(molecules[i])
					--print(transmitters[j])
				end
			end
		end
	end
end



-- The main functionality
readConfiguration() -- Read configuration

-- For the first iteration of this project the number of transmitters and receivers are assumed to be 1.
dimension = configData[indexDimension]
runTime = configData[indexRunTime]
deltaTime = configData[indexDeltaTime]
molecules = torch.CudaTensor(configData[indexMolecule], 3): uniform(0,dimension)
moleculeAvailability = torch.CudaTensor(configData[indexMolecule]): fill(1)

transmitters = torch.CudaTensor(1, 3)
receivers = torch.CudaTensor(1, 3)


-- execution
transmitters[1] = torch.CudaTensor({1,1,1})
receivers[1] = torch.CudaTensor({9, 9, 9})
mean = 0.1
stdev = 1

numberOfMoleculesReceived = 0

for time = 0, runTime, deltaTime do
	--cutorch.reserveStreams(configData[indexMolecule])
	--local streamList = {}
	for i = 1, configData[indexMolecule] do
		 --cutorch.setStream(i)
   		 --streamList[i] = i
		if moleculeAvailability[i] == 1 then
			molecules[i][1] = molecules[i][1] + torch.normal(mean, stdev)
			molecules[i][2] = molecules[i][2] + torch.normal(mean, stdev)
			molecules[i][3] = molecules[i][3] + torch.normal(mean, stdev)
		end
	end
	check()
	--set back the stream to default stream (0):
	--cutorch.setDefaultStream()

	-- 0 is default stream, let 0 wait for the n streams to complete before doing anything further
	--cutorch.streamWaitFor(0, streamList)
end

print('The number of received molecules:' .. numberOfMoleculesReceived)

























