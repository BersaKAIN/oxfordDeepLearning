require 'requ'

-- NOTE: Assumes input and output to module are 1-dimensional, i.e. doesn't test the module
--       in mini-batch mode. It's easy to modify it to do that if you want, though.
local function jacobian_wrt_input(module, x, eps)
  -- compute true Jacobian (rows = over outputs, cols = over inputs, as in our writeup's equations)
  local z = module:forward(x):clone()
  local jac = torch.DoubleTensor(z:size(1), x:size(1))
  
  -- get true Jacobian, ROW BY ROW
  local one_hot = torch.zeros(z:size())
  for i = 1, z:size(1) do
    one_hot[i] = 1
    jac[i]:copy(module:backward(x, one_hot))
    one_hot[i] = 0
  end
  
  -- compute finite-differences Jacobian, COLUMN BY COLUMN
  local jac_est = torch.DoubleTensor(z:size(1), x:size(1))
  for i = 1, x:size(1) do
    -- TODO: modify this to perform a two-sided estimate. Remember to do this carefully, because 
    --       nn modules reuse their output buffer across different calls to forward.
    -- note that z_offsets_plus and z_offsets_minus both points to the state output of the module so they are always the same.
    x[i] = x[i] + eps
    local z_offset_plus = module:forward(x)
    x[i] = x[i] - 2*eps
    jac_est[{{},i}]:copy(z_offset_plus):add(-1, module:forward(x)):div(2*eps) -- double sided
    -- local z_offset_minus = module:forward(x)
    x[i] = x[i] + eps -- set x back to normal

    -- jac_est[{{},i}]:copy(z_offset_plus):add(-1, z_offset_minus):div(2*eps) -- double sided
    -- jac_est[{{},i}]:copy(z_offset_plus):add(-1, z):div(eps) -- single sided
  end

  -- computes (symmetric) relative error of gradient
  local abs_diff = (jac - jac_est):abs()
  return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

---------------------------------------------------------
-- test our layer in isolation
--
torch.manualSeed(1)
local requ = nn.ReQU()

local x = torch.randn(10) -- random input to layer
print(x)
print(jacobian_wrt_input(requ, x, 1e-6))

