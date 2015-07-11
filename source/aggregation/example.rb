require './init'

sessions = ExampleSession.load

Pipeline.new(
    :output_dir => 'example-dir',
    :normalize => false,
    :flow => {[ActionUnitProcessor] => [AverageAggregator] }
).run(sessions)