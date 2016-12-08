cutorch = require 'cutorch'

simulationRunTime = 1000
numberOfMolecules = 10000000
molecules = torch.CudaTensor(3, numberOfMolecules): uniform(0, 10)
availability = torch.CudaTensor(numberOfMolecules): fill(1)
for i = 1, simulationRunTime do
	delta1 = torch.CudaTensor(numberOfMolecules): normal(0, 10)
	delta2 = torch.CudaTensor(numberOfMolecules): normal(0, 10)  
	delta3 = torch.CudaTensor(numberOfMolecules): normal(0, 10) 
	molecules[1]: add(delta1:cmul(availability))
	molecules[2]: add(delta2:cmul(availability))
	molecules[3]: add(delta3:cmul(availability))
	-- single receiver.
	dd1 = molecules[1]:csub(2)
	dd2 = molecules[2]:csub(3)
	dd3 = molecules[3]:csub(4)

	sq1 = torch.cmul(dd1, dd1)
	sq2 = torch.cmul(dd2, dd2)
	sq3 = torch.cmul(dd3, dd3)
	
	dist = sq1:add(sq2)
	dist:add(sq3)
	dist:pow(0.5)
	availability = dist:le(3)
 			
end
