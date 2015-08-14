require 'nn'

-- torch.class(newClass, parentClass)
local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input) -- copy is making a deepcopy (creating new memory for that instance)
  -- ge(0) check element-wise greater or equal to 0 and returns a byte tensor. double() will convert it to double.
  self.output:cmul(self.output:ge(0):double()):cmul(self.output)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:cmul(input:ge(0):double()):cmul(input):mul(2)
  -- ...something here...
  return self.gradInput
end	

