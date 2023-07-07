%% Options
clear

correlationAlgorithmValidation = true;
fieldAlgorithmValidation = true;

%% Configuration
% spatial - in cm
% temporal - in seconds

mcxData = load(['examples',filesep,'data',filesep,'mcxValidationData.mat']);

% Simulation
config.simulation.batchSize = 10000;
config.simulation.deviceNumber = 0;
config.simulation.lambda = 500e-9 * 100; % 500 nm in cm

% Scattering
% All scatterers have isotropic scattering
config.scattering.type = repmat({'HG'},1,3);
config.scattering.amplitudeFunction = cell(1,3);

% material {1} is outside the volume, usually unused.
config.scattering.amplitudeFunction{1} = 0;

% material {2} is static part of volume.
config.scattering.amplitudeFunction{2} = 0;

% material {3} is dynamic part of volume.
config.scattering.amplitudeFunction{3} = 0;

config.medium.type = 'Heterogeneous';
% dims of the box volume, in this case:
% mcxData.dims(1) x mcxData.dims(2) x mcxData.dims(3)
% xz axis are divided to grid, which we set as static / dynamic
% The division is not forced to be equal
% R - dynmic part size
% h - distance between dynamic parts
config.medium.xAxis = -mcxData.dims(1)/2:mcxData.h/2:mcxData.dims(1)/2;
if(config.medium.xAxis(end) ~= mcxData.dims(1)/2)
    config.medium.xAxis(end + 1) = mcxData.dims(1)/2;
end
if(numel(config.medium.xAxis(2:2:end)) == config.medium.xAxis(1:2:end))
    config.medium.xAxis(2:2:end) = config.medium.xAxis(1:2:end) + mcxData.R;
else
    config.medium.xAxis(2:2:end) = config.medium.xAxis(1:2:(end-1)) + mcxData.R;
end

config.medium.yAxis = [-mcxData.dims(2)/2,mcxData.dims(2)/2];

config.medium.zAxis = 0:mcxData.h/2:mcxData.dims(3);
if(config.medium.zAxis(end) ~= mcxData.dims(3))
    config.medium.zAxis(end + 1) = mcxData.dims(3);
end
if(numel(config.medium.zAxis(2:2:end)) == config.medium.zAxis(1:2:end))
    config.medium.zAxis(2:2:end) = config.medium.zAxis(1:2:end) + mcxData.R;
else
    config.medium.zAxis(2:2:end) = config.medium.zAxis(1:2:(end-1)) + mcxData.R;
end

% Material grid is in size of:
% (numel(xAxis) - 1) x (numel(yAxis) - 1) x (numel(zAxis) - 1)
% each entry defines the material type of a voxel

% we set 1 for static parts
config.medium.materialGrid = ones(numel(config.medium.xAxis) - 1, ...
    numel(config.medium.yAxis) - 1, numel(config.medium.zAxis) - 1);

% we set value 2 for dynamic parts
config.medium.materialGrid(1:2:end,:,1:2:end) = 2;

% define the MFP in [outside, static, dynamic]
sigs = [0, 1/mcxData.MFP, 1/mcxData.MFP];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

% Illumination
config.illumination.type = 'PointSource';
config.illumination.location.x = -mcxData.viSep/2;
config.illumination.location.y = 0;
config.illumination.location.z = 0;

% View
config.view.type = 'PointSource';
config.view.location.x = mcxData.viSep/2;
config.view.location.y = 0;
config.view.location.z = 0;

% Sampler
config.sampler.t = mcxData.t;

% Brownian motion in [outside, static, dynamic]
config.sampler.D = [0,0,mcxData.D];

% Assuming no linear motion
config.sampler.U.x = [0,0,0];
config.sampler.U.y = [0,0,0];
config.sampler.U.z = [0,0,0];

% Tracer
config.tracer.type = 'nee';

%% Run correlation
if(correlationAlgorithmValidation)

    config.simulation.fullIteration = 1000;
    config.simulation.renderingsNum = 1;
    config.sampler.type = 'TemporalCorrelationSampler';


    % We model temporal only, thus i1 = i2 and v1 = v2
    config.illumination2 = config.illumination;
    config.view2 = config.view;

    tic
    [u, iter] = sstmc3D(config);
    runTimeCorrelation = toc

    C_correlation = u / iter;
end

%% Run field
if(fieldAlgorithmValidation)

    config.simulation.fullIteration = 1;
    config.simulation.renderingsNum = 1000;
    config.sampler.type = 'TemporalSampler';

    tic
    [u, iter] = sstmc3D(config);
    runTimeField = toc

    C_field = mean(u .* conj(u(1,1,1,:)) ./ permute(iter,[1,3,4,2]),4);
end

%% Plot

lgd = {'mcx'};

if(correlationAlgorithmValidation)
    lgd = [lgd, 'MC correlation'];
end

if(fieldAlgorithmValidation)
    lgd = [lgd, 'MC field'];
end

figure
hold on

plot(config.sampler.t, mcxData.mcx_res);

if(correlationAlgorithmValidation)
    plot(config.sampler.t, abs(C_correlation(:)));
end

if(fieldAlgorithmValidation)
    plot(config.sampler.t, abs(C_field(:)));
end

legend(lgd)

xlabel('time (s)')
ylabel('temporal speckle correlation')
title('MCX v.s. spatio-temporal MC')

