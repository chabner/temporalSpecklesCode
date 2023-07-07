%% Configuration
% spatial - in cm
% temporal - in seconds

% Simulation
config.simulation.batchSize = 500;
config.simulation.fullIteration = 10;
config.simulation.renderingsNum = 1;
config.simulation.deviceNumber = 0;
config.simulation.lambda = 500e-9 * 100; % 500 nm in cm

% Scattering
% All scatterers have isotropic scattering
% Assume homogeneous medium
config.scattering.type = repmat({'HG'},1,2);
config.scattering.amplitudeFunction = cell(1,2);

% material {1} is outside the volume, usually unused.
config.scattering.amplitudeFunction{1} = 0;

% material {2} is inside the volume.
config.scattering.amplitudeFunction{2} = 0;

config.medium.type = 'Heterogeneous';
config.medium.xAxis = [-2,2];
config.medium.yAxis = [-2,2];
config.medium.zAxis = [0,2];

config.medium.materialGrid = 1;

% define the MFP in [outside, static, dynamic]
sigs = [0, 1/0.1];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

xAxis = (-1:0.04:1) * 1e-4;

% Illumination
config.illumination.type = 'PointSource';
config.illumination.location.x = -0.1 + xAxis(1) + 0 * xAxis;
config.illumination.location.y = 0 * xAxis;
config.illumination.location.z = 0 * xAxis;

config.illumination2.type = 'PointSource';
config.illumination2.location.x = -0.1 + xAxis;
config.illumination2.location.y = 0 * xAxis;
config.illumination2.location.z = 0 * xAxis;

% View
config.view.type = 'PointSource';
config.view.location.x = 0.1 + xAxis(1) + 0 * xAxis;
config.view.location.y = 0 * xAxis;
config.view.location.z = 0 * xAxis;

config.view2.type = 'PointSource';
config.view2.location.x = 0.1 + xAxis;
config.view2.location.y = 0 * xAxis;
config.view2.location.z = 0 * xAxis;

% Sampler
config.sampler.type = 'TemporalCorrelationSampler';
config.sampler.t = (0:1:50)*1e-6;

% Brownian motion in [outside, inside]
config.sampler.D = [0,1e-6];

% Assuming no linear motion
config.sampler.U.x = [0,4];
config.sampler.U.y = [0,0];
config.sampler.U.z = [0,0];

% Tracer
config.tracer.type = 'nee';

tic
[u, iter] = sstmc3D(config);
runTimeCorrelation = toc

C = u / iter;

%%
C_noTracking = permute(abs(C(1,1,:)),[3,2,1]);
C_tracking = zeros(1,51);


for ii = 1:1:51
    C_tracking(ii) = abs(C(ii,ii,ii));
end

figure
hold on
plot(config.sampler.t,C_noTracking)
plot(config.sampler.t,C_tracking)

title('Tracking v.s. without tracking')
xlabel('time (sec)')
ylabel('Speckle correlation')
legend('Without tracking','With tracking')
