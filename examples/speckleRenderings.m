%% Options
clear

sourceType = 'point';
% sourceType = 'directional';
% sourceType = 'focused';

configurationType = 'transmissive';
% configurationType = 'backscattering';

pixelsNum = 81;
dp = 0.2;

%% Configuration

% Simulation
config.simulation.batchSize = 1;
config.simulation.deviceNumber = 0;
config.simulation.fullIteration = 100;
config.simulation.lambda = 1;

% Scattering
% All scatterers have isotropic scattering
% for this example we use isotropic scattering

if(strcmp(sourceType,'focused') == 1)
    % focused sources are only working with vMF mixture
    config.scattering.type = repmat({'vMF'},1,2);
    config.scattering.amplitudeFunction = cell(1,2);

    config.scattering.amplitudeFunction{1}.mixtureMu = 0;
    config.scattering.amplitudeFunction{1}.mixtureC = -1.2655121234846453964889;
    config.scattering.amplitudeFunction{1}.mixtureAlpha = 1;

    config.scattering.amplitudeFunction{2}.mixtureMu = 0;
    config.scattering.amplitudeFunction{2}.mixtureC = -1.2655121234846453964889;
    config.scattering.amplitudeFunction{2}.mixtureAlpha = 1;
else
    config.scattering.type = repmat({'HG'},1,2);
    config.scattering.amplitudeFunction = cell(1,2);

    % material {1} is outside the volume, usually unused.
    config.scattering.amplitudeFunction{1} = 0;

    % material {2} is inside the volume
    config.scattering.amplitudeFunction{2} = 0;

end

config.medium.type = 'Heterogeneous';
config.medium.xAxis = [-1000,1000];
config.medium.yAxis = [-1000,1000];
config.medium.zAxis = [0,100];
config.medium.materialGrid = 1;

% define the MFP in [outside, inside]
sigs = [0, 1/40];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

if(strcmp(sourceType,'point') == 1)
    % Illumination
    config.illumination.type = 'PointSource';
    config.illumination.location.x = (-2:1:2) * dp;
    config.illumination.location.y = zeros(1,5);
    config.illumination.location.z = -10 * ones(1,5);

    % View
    [X,Y] = ndgrid((-(pixelsNum-1)/2:1:(pixelsNum-1)/2)*dp);
    config.view.type = 'PointSource';
    config.view.location.x = X(:);
    config.view.location.y = Y(:);

    if(strcmp(configurationType,'transmissive') == 1)
        config.view.location.z = 110 + 0 * X(:);
    else
        config.view.location.z = -10 + 0 * X(:);
    end
elseif(strcmp(sourceType,'directional') == 1)
    % Illumination
    config.illumination.type = 'FarField';
    config.illumination.direction.x = deg2rad(-2:1:2) * dp / 20;
    config.illumination.direction.y = zeros(1,5);
    config.illumination.direction.z = sqrt(1-config.illumination.direction.x.^2);

    % View
    [X,Y] = ndgrid(deg2rad((-(pixelsNum-1)/2:1:(pixelsNum-1)/2)*dp) / 20);
    config.view.type = 'FarField';
    config.view.direction.x = X(:);
    config.view.direction.y = Y(:);

    if(strcmp(configurationType,'transmissive') == 1)
        config.view.direction.z = sqrt(1-config.view.direction.x.^2-config.view.direction.y.^2);
    else
        config.view.direction.z = -sqrt(1-config.view.direction.x.^2-config.view.direction.y.^2);
    end
elseif(strcmp(sourceType,'focused') == 1)
    % Illumination
    config.illumination.type = 'GaussianBeam';
    config.illumination.focalPoint.x = (-2:1:2) * dp;
    config.illumination.focalPoint.y = zeros(1,5);
    config.illumination.focalPoint.z = zeros(1,5);

    config.illumination.focalDirection.x = zeros(1,5);
    config.illumination.focalDirection.y = zeros(1,5);
    config.illumination.focalDirection.z = ones(1,5);

    config.illumination.aperture = 0.4;

    % View
    [X,Y] = ndgrid((-(pixelsNum-1)/2:1:(pixelsNum-1)/2)*dp);
    config.view.type = '    ';
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
end

% Sampler
config.sampler.type = 'TemporalSampler';
config.sampler.t = 0:0.25:1;

% Brownian motion in [outside, inside]
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
u = permute(reshape(u,5,pixelsNum,pixelsNum,5),[3,2,1,4]);

%%
maxVal = max(abs(u(:)));

figure
subplot(2,5,1)
imagesc(abs(u(:,:,1,1)),[0,maxVal]);
title('u^1(t_1)')
subplot(2,5,2)
imagesc(abs(u(:,:,2,1)),[0,maxVal]);
title('u^2(t_1)')
subplot(2,5,3)
imagesc(abs(u(:,:,3,1)),[0,maxVal]);
title('u^3(t_1)')
subplot(2,5,4)
imagesc(abs(u(:,:,4,1)),[0,maxVal]);
title('u^4(t_1)')
subplot(2,5,5)
imagesc(abs(u(:,:,5,1)),[0,maxVal]);
title('u^5(t_1)')

subplot(2,5,6)
imagesc(abs(u(:,:,1,1)),[0,maxVal]);
title('u^1(t_1)')
subplot(2,5,7)
imagesc(abs(u(:,:,1,2)),[0,maxVal]);
title('u^2(t_2)')
subplot(2,5,8)
imagesc(abs(u(:,:,1,3)),[0,maxVal]);
title('u^3(t_3)')
subplot(2,5,9)
imagesc(abs(u(:,:,1,4)),[0,maxVal]);
title('u^4(t_4)')
subplot(2,5,10)
imagesc(abs(u(:,:,1,5)),[0,maxVal]);
title('u^5(t_5)')

colormap hot