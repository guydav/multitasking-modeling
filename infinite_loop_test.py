import psyneulink as pnl


def create_weights(in_size=2, out_size=2):
    return pnl.random_matrix(in_size, out_size, 2, -1) * 0.1

# Create the layers
input_layer = pnl.TransferMechanism(size=2, name='input')
output_layer = pnl.TransferMechanism(size=2, name='output')
indirect_layer = pnl.TransferMechanism(size=2, name='indirect')

"""
If process block #1 is before process block #2, the system creation hangs in an infinite warning loop
However, if process block #2 is before process block #1, the system creates just fine.
"""

# Process block #1 -- Create the input-output process
input_output_process = pnl.Process(pathway=[input_layer, create_weights(), output_layer],
                                   name='input-output', learning=pnl.LEARNING)
input_output_process.pathway[1].learning_mechanism.learning_rate = 0.1

# Process block #2 -- create the indirect processes
input_indirect_process = pnl.Process(pathway=[input_layer, create_weights(), indirect_layer],
                                     name='input-indirect', learning=pnl.LEARNING)
input_indirect_process.pathway[1].learning_mechanism.learning_rate = 0.5

indirect_output_process = pnl.Process(pathway=[indirect_layer, create_weights(), output_layer],
                                     name='indirect-output', learning=pnl.LEARNING)
indirect_output_process.pathway[1].learning_mechanism.learning_rate = 0.5


# Create the system
system = pnl.System(name='infinite loop test', processes=[input_output_process,
                                                          input_indirect_process,
                                                          indirect_output_process])

print(system)




