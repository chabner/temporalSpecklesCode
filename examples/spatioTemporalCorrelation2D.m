%% Options
pixelsNum = 51;
% motionType = 'Brownian';
motionType = 'Brownian-Linear';
% motionType = 'Linear';

sourceViewSeperation = 80;
distanceFromEdgeIllumination = 30;
distanceFromEdgeView = 30;
MFP = 40;

Ux = 1;
D = 0.015;

%% Configuration
load(['examples',filesep,'data',filesep,'inPhasefunction.mat'])

% Simulation
config.simulation.batchSize = 100;
config.simulation.fullIteration = 100;
config.simulation.deviceNumber = 0;
config.simulation.lambda = 1;

% Scattering
config.scattering.type = repmat({'Tabular'},1,2);
config.scattering.amplitudeFunction = cell(1,2);

config.scattering.amplitudeFunction{1} = mudiffPhaseFunction;
config.scattering.amplitudeFunction{2} = mudiffPhaseFunction;

config.medium.type = 'Heterogeneous';
config.medium.xAxis = [-50, 50];
config.medium.yAxis = [-20, 20];
config.medium.materialGrid = 1;

sigt = [0, 1/MFP];
config.medium.sigs = sigt;
config.medium.siga = sigt * 0;

% Illumination
x_pix = -1:(2/(pixelsNum-1)):1;

config.illumination.type = 'PointSource';
config.illumination.location.x = 0 * x_pix + x_pix(1) - sourceViewSeperation/2;
config.illumination.location.y = 0 * x_pix - distanceFromEdgeIllumination;

config.illumination2.type = 'PointSource';
config.illumination2.location.x = x_pix - sourceViewSeperation/2;
config.illumination2.location.y = 0 * x_pix - distanceFromEdgeIllumination;

% View

config.view.type = 'PointSource';
config.view.location.x = 0 * x_pix + x_pix(1) + sourceViewSeperation/2;
config.view.location.y = 0 * x_pix - distanceFromEdgeView;

config.view2.type = 'PointSource';
config.view2.location.x = x_pix + sourceViewSeperation/2;
config.view2.location.y = 0 * x_pix - distanceFromEdgeView;

% Sampler
config.sampler.type = 'TemporalCorrelationSampler';
config.sampler.t = 0:(2/(pixelsNum-1)):2;

if(strcmp(motionType,'Linear') == 0)
    config.sampler.D = [0,D];
else
    config.sampler.D = [0,0];
end

if(strcmp(motionType,'Brownian') == 0)
    config.sampler.U.x = [0,Ux];
else
    config.sampler.U.x = [0,0];
end

config.sampler.U.y = [0,0];

% Tracer
config.tracer.type = 'nee';

%% run
tic
[u, iter] = sstmc2D(config);
runTime = toc

C = u / iter;

%% Extract

C_noTracking = permute(C(1,:,:),[2,3,1]);

C_tracking = zeros(pixelsNum,pixelsNum);

for ii = 1:1:pixelsNum
    C_tracking(:,ii) = diag(C(:,:,ii));
end


%% Plot
f = figure;
f.Position = [357,204,1185,671];
subplot(1,2,1)
imagesc(abs(C_noTracking))
title([motionType,' motion spatio-temporal correlation without tracking'])

subplot(1,2,2)
imagesc(abs(C_tracking))
title([motionType,' motion spatio-temporal correlation with tracking'])