require './init'

sessions = MimeSession.load

Pipeline.new(
    :output_dir => 'mime-koelstra',
    :normalize => false,
    :flow => {[EmfacsActionUnitProcessor] => []}
).run(sessions)