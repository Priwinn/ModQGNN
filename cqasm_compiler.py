import openql as ql

def cqasm_compiler(cqasm_file,qbits):
    new_scheduler='yes'
    scheduler='ASAP'
    uniform_sched= 'no'
    sched_commute = 'yes'
    mapper='minextend'
    moves='yes'
    maptiebreak='first'
    initial_placement='no'
    output_dir_name='random_output'
    optimize='no'
    measurement=True
    log_level='LOG_WARNING'

    output_dir = 'compiler_output'
    ql.initialize()
    # uses defaults of options in mapper branch except for output_dir and for maptiebreak
    ql.set_option('output_dir', output_dir)     # this uses output_dir set above
    ql.set_option('maptiebreak', maptiebreak)       # this makes behavior deterministic to cmp with golden
    # and deviates from default

    ql.set_option('log_level', log_level)
    ql.set_option('optimize', optimize)
    ql.set_option('mapmaxalters', '10')
    ql.set_option('scheduler_heuristic', 'random')
    ql.set_option('use_default_gates', 'no')
    ql.set_option('generate_code', 'no')

    ql.set_option('decompose_toffoli', 'no')
    ql.set_option('scheduler', scheduler)
    ql.set_option('scheduler_uniform', uniform_sched)
    ql.set_option('scheduler_commute', sched_commute)
    ql.set_option('scheduler_commute_rotations', 'yes')
    ql.set_option('prescheduler', 'yes')
    ql.set_option('cz_mode', 'manual')
    ql.set_option('print_dot_graphs', 'no')

    ql.set_option('clifford_premapper', 'yes')
    ql.set_option('clifford_postmapper', 'no')
    ql.set_option('mapper', 'no')
    ql.set_option('mapinitone2one', 'yes')
    ql.set_option('mapassumezeroinitstate', 'yes')
    ql.set_option('initialplace', initial_placement)
    ql.set_option('initialplace2qhorizon', '0')
    ql.set_option('mapusemoves', moves)
    ql.set_option('mapreverseswap', 'yes')
    ql.set_option('mappathselect', 'random')
    ql.set_option('maplookahead', 'noroutingfirst')
    ql.set_option('maprecNN2q', 'no')
    ql.set_option('mapselectmaxlevel', '0')
    ql.set_option('mapselectmaxwidth', 'min')
    #ql.set_option('scheduler_post179', new_scheduler)
    ql.set_option('write_report_files', 'no')

    # platform  = ql.Platform('platform_none', config_file)
    platform  = ql.Platform('starmon', '/home/ruizhe/Code/config_quantum/test_mapper_100_not_constrained.json')
    p = ql.Program(cqasm_file.split('/')[-1].strip('.qasm'), platform, qbits)
    p.get_compiler().prefix_pass('io.cqasm.Read','read',{'cqasm_file':cqasm_file})

    p.compile()