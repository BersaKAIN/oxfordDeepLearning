---------------------------------------------------------------------------------------
-- Practical 5 - Learning to use nngraph to build complex neural networks archietecture
--
-- to run: 
---------------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nngraph'
require 'graph'

-- nngraph overloads the call operator (i.e. () operator used for function calls) on all
-- nn.Module objects. It will return a node that wraps the nn.Module. The call operator
-- will take the nodes parents.
-- eg: nn.Module(<arguments_of_nn.Module>)(<parent_of_the_node>)

-- then we use nn.gModule to create a module taking some nodes in the graph to be inputs
-- and outputs.
-- eg: nn.gModule(<table_of_inputs>,<table_of_outputs>)


-- params for the linear layer
params = {
	x3_size1 = 10,
	x3_size2 = 30
}
-- dummy nodes to take input data as nodes in graph
x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()

-- modeling output = x1 + x2 cmul linear(x3)
l3 = nn.Linear(params.x3_size1, params.x3_size2)(x3)
m23 = nn.CMulTable()({x2,l3})
add = nn.CAddTable()({x1, m23})

-- specify the inputs and outputs of the graph
m = nn.gModule({x1,x2,x3}, {add})
