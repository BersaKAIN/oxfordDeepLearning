-- Coding problem for oxford deep learning course problems set 2
-- Yiran Zhang

-- Movie Data
movieData = torch.Tensor({{0,0,-1,0,-1,1,1}, {-1,1,1,-1,0,1,1}, {0,1,1,0,0,-1,1}, {-1,1,1,0,0,1,1}, {0,1,1,0,0,1,1}, {1,-1,1,1,1,-1,0}, {-1,1,-1,0,-1,0,1}, {0,-1,0,1,1,-1,-1}, {0,0,-1,1,1,0,-1}})

-- Params
reg = 0.1
f = 2
dims = movieData:size()
m = dims[1] -- m users
n = dims[2] -- n movies

-- Random initialization
X = torch.rand(m,f) -- generates a tensor of dimension m,f
Y = torch.rand(f,n)

X:mul(-2):add(1):mul(0.1) -- element-wise operation for each of the tensor
Y:mul(-2):add(1):mul(0.1)

C = torch.abs(movieData) -- get the absulote value of movieData and stores in another variable C

for i=1,100 do
	-- solve X while Y being fixed
	for j=1,m do
		Cu = torch.diag(C[j])
		-- general equation solver for AX = B -> X = torch.gesv(B,A)
		-- however if A is upper triangular, we should use torch.trtrs
		-- view() is changing tensor representation of the tensor without changing its memory allocation. Here we are chaning the 1D tensor into a 2D tensor with size 1 in the second dimension.
		-- -1 as the first parameter in the view function will make an auto guess of the dimension based on the information from other dimensions.
		X[j] = torch.gesv(
			(Y * Cu * movieData[j]):view(-1,1),
			Y * Cu * Y:t() + torch.eye(f):mul(reg)
			)
	end
	-- solve Y while X being fixed
	for j=1,n do
		Ci = torch.diag(C:select(2,j)) -- select the jth column, select() takes two parameters, the first is which dimension we are slicing on the second is which number should the selected dim be.
		-- bbb = X:t() * Ci * movieData:select(2,j)
		-- aaa = X:t() * Ci * X + torch.eye(f):mul(reg)
		-- Y:t()[j] = torch.gesv(bbb:view(-1,1),aaa)
		-- Y:select(2,j) = torch.gesv( (X:t() * Ci * movieData:select(2,j)):view(-1,1), X:t() * Ci * X + torch.eye(f):mul(reg) )
		Y:t()[j] = torch.gesv(
			(X:t() * Ci * movieData:select(2,j)):view(-1,1),
			X:t() * Ci * X + torch.eye(f):mul(reg)
			)
	end
end

print(X)
print(Y)
print(X*Y)


