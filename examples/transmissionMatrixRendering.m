%% Options
clear

% configurationType = 'transmissive';
configurationType = 'backscattering';

pixelsNum = 51;
dp = 0.2;

%% Configuration

% Simulation
config.simulation.batchSize = 1;
config.simulation.deviceNumber = 0;
config.simulation.fullIteration = 1000;
config.simulation.lambda = 1;

% Scattering
% All scatterers have isotropic scattering
% for this example we use isotropic scattering

% focused sources are only working with vMF mixture
config.scattering.type = repmat({'vMF'},1,2);
config.scattering.amplitudeFunction = cell(1,2);

config.scattering.amplitudeFunction{1}.mixtureMu = 0;
config.scattering.amplitudeFunction{1}.mixtureC = -1.2655121234846453964889;
config.scattering.amplitudeFunction{1}.mixtureAlpha = 1;

config.scattering.amplitudeFunction{2}.mixtureMu = 0;
config.scattering.amplitudeFunction{2}.mixtureC = -1.2655121234846453964889;
config.scattering.amplitudeFunction{2}.mixtureAlpha = 1;

config.medium.type = 'Heterogeneous';
config.medium.xAxis = [-1000,1000];
config.medium.yAxis = [-1000,1000];
config.medium.zAxis = [0,100];
config.medium.materialGrid = 1;

% define the MFP in [outside, inside]
sigs = [0, 1/40];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

[X,Y] = ndgrid((-(pixelsNum-1)/2:1:(pixelsNum-1)/2)*dp);

% Illumination
config.illumination.type = 'GaussianBeam';
config.illumination.focalPoint.x = X(:);
config.illumination.focalPoint.y = Y(:);
config.illumination.focalPoint.z = zeros(size(X(:)));

config.illumination.focalDirection.x = zeros(size(X(:)));
config.illumination.focalDirection.y = zeros(size(X(:)));
config.illumination.focalDirection.z = ones(size(X(:)));

config.illumination.aperture = 0.4;

% View
config.view.type = 'GaussianBeam';
config.view.focalPoint.x = X(:);
config.view.focalPoint.y = Y(:);

config.view.focalDirection.x = zeros(size(X(:)));
config.view.focalDirection.y = zeros(size(X(:)));

if(strcmp(configurationType,'transmissive') == 1)
    config.view.focalPoint.z = 100 + 0 * X(:);
    config.view.focalDirection.z = ones(size(X(:)));
else
    config.view.focalPoint.z = 0 * X(:);
    config.view.focalDirection.z = -ones(size(X(:)));
end

config.view.aperture = 0.4;

% Sampler
config.sampler.type = 'TemporalSampler';
config.sampler.t = 0:0.25:1;

% Brownian motion in [outside, static]
config.sampler.D = [0,1e-3];

% Assuming no linear motion
config.sampler.U.x = [0,0];
config.sampler.U.y = [0,0];
config.sampler.U.z = [0,0];

% Tracer
config.tracer.type = 'nee';

tic
[u, iter] = sstmc3D(config);
runTimeCorrelation = toc

u = u / sqrt(iter);

%%
maxVal = max(abs(u(:)));

figure
subplot(1,5,1)
imagesc(abs(u(:,:,1)),[0,maxVal]);
title('TM(t_1)')
subplot(1,5,2)
imagesc(abs(u(:,:,2)),[0,maxVal]);
title('TM(t_2)')
subplot(1,5,3)
imagesc(abs(u(:,:,3)),[0,maxVal]);
title('TM(t_3)')
subplot(1,5,4)
imagesc(abs(u(:,:,4)),[0,maxVal]);
title('TM(t_4)')
subplot(1,5,5)
imagesc(abs(u(:,:,5)),[0,maxVal]);
title('TM(t_5)')

colormap hot