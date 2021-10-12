#Hands-on excercize

# Import the OpenQL module:

import openql as ql
import os

# Initialize() function – to reinitialize OpenQl to default configuration

ql.initialize()

# Changing global options (e.g. to define output directory location or log level):

ql.set_option('output_dir', 'OpenQl Tutorial/output')
ql.set_option('log_level', 'LOG_INFO')
ql.set_option('write_report_files', 'yes')


# Adding compiler related options:

#check available options:

ql.print_options()


# scheduling
ql.set_option("scheduler", "ALAP")
ql.set_option('scheduler_commute', 'yes')
ql.set_option('scheduler_commute_rotations', 'yes')
ql.set_option('prescheduler', 'yes')

# initial placement

ql.set_option('initialplace', 'no') 
ql.set_option('mapinitone2one', 'yes')

# optimization and routing

ql.set_option('clifford_premapper', 'yes')
ql.set_option('mapper','minextend')
ql.set_option('clifford_postmapper', 'no')
ql.set_option('mapassumezeroinitstate', 'yes')
ql.set_option('maplookahead', 'noroutingfirst')
ql.set_option('mappathselect', 'all')
ql.set_option('mapselectswaps','all')
ql.set_option('maptiebreak', 'random')
ql.set_option('mapusemoves', 'no')


# Setting up platform (‘none’ is very basic platform configuration good for simulations with QX ):

platform = ql.Platform('my_platform', 'test_mapper_5Q.json')

# Setting up number of qubits, program nad kernel name:

nqubits = 5
program = ql.Program('my_program', platform, nqubits)
k = ql.Kernel('my_kernel', platform, nqubits)

# Adding gates to kernel for creating circuit:

k.gate('ym90', (0,))
k.gate('x45', (1,))
k.gate('x45', (2,))
k.gate('x', (3,))
k.gate('y45', (4,))
k.gate('cz', (0, 4))
k.gate('y90', (2,))
k.gate('x45', (0,))
k.gate('cz', (0, 1))
k.gate('cz', (4, 1))
k.gate('xm90', (0,))
k.gate('cz', (3, 0))
k.gate('xm90', (1,))
k.gate('x90', (4,))
k.gate('cz', (3, 2))

# Adding kernel to program:

program.add_kernel(k)

#visualize circuit before and after mapping

curdir = os.path.dirname(__file__)
output_dir = os.path.join(curdir, 'visualizer_output')

program.get_compiler().insert_pass_before(
    'mapper',
    'ana.visualize.Mapping',
    'before_mapping', {
        'config': os.path.join(curdir, "visualizer_config_example1.json"),
        'output_prefix': output_dir + '\%N_before',
        'interactive': 'yes'
    }
)

program.get_compiler().insert_pass_after(
    'mapper',
    'ana.visualize.Mapping',
    'after_mapping', {
        'config': os.path.join(curdir, "visualizer_config_example1.json"),
        'output_prefix': output_dir + '\%N_after',
        'interactive': 'yes'
    }
)

# Compiling program:

program.compile()