require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'au-counts',
    :normalize => false,
    :flow => {[ActionUnitProcessor] => [AverageAggregator] }
).run(sessions)