---------------------------------------------------------------------------------------
-- Practical 5 - Learning to use nngraph to build complex neural networks archietecture
--
-- to run: 
---------------------------------------------------------------------------------------

require 'torch'
require 'nn'

-- nngraph overloads the call operator (i.e. () operator used for function calls) on all
-- nn.Module objects. It will return a node that wraps the nn.Module. The call operator
-- will take the nodes parents.
-- eg: nn.Module(<arguments_of_nn.Module>)(<parent_of_the_node>)

-- then we use nn.gModule to create a module taking some nodes in the graph to be inputs
-- and outputs.
-- eg: nn.gModule(<table_of_inputs>,<table_of_outputs>)

