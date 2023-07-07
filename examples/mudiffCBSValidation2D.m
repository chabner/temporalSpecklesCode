%% Options
clear

correlationAlgorithmValidation = true;
fieldAlgorithmValidation = true;

%% Configuration
mudiffData = load(['examples',filesep,'data',filesep,'mudiffCBS.mat']);

% Simulation
config.simulation.batchSize = 1000;
config.simulation.deviceNumber = 0;
config.simulation.lambda = 1;

% Scattering
config.scattering.type = repmat({'Tabular'},1,2);
config.scattering.amplitudeFunction = cell(1,2);

config.scattering.amplitudeFunction{1} = mudiffData.ampFunc;
config.scattering.amplitudeFunction{2} = mudiffData.ampFunc;


config.medium.type = 'Heterogeneous';
config.medium.xAxis = mudiffData.x;
config.medium.yAxis = mudiffData.y;
config.medium.materialGrid = 1;

% define the MFP in [outside, inside]
sigs = [0, 1/mudiffData.MFP];
config.medium.sigs = sigs;
config.medium.siga = sigs * 0;

% Illumination
config.illumination.type = 'FarField';
config.illumination.direction.x = 0;
config.illumination.direction.y = 1;

% View
config.view.type = 'FarField';
config.view.direction.x = sin(mudiffData.degList);
config.view.direction.y = cos(mudiffData.degList);

% % Illumination
% config.illumination.type = 'FarField';
% config.illumination.location.x = 0;
% config.illumination.location.y = 1;
% 
% % View
% config.view.type = 'FarField';
% config.view.location.x = sin(mudiffData.degList);
% config.view.location.y = cos(mudiffData.degList);

% Sampler
config.sampler.t = 0;

% Brownian motion in [outside, inside]
config.sampler.D = [0,0];

% Assuming no linear motion
config.sampler.U.x = [0,0];
config.sampler.U.y = [0,0];

% Tracer
config.tracer.type = 'nee';

%% Run correlation
if(correlationAlgorithmValidation)

    config.simulation.fullIteration = 100;
    config.simulation.renderingsNum = 1;
    config.sampler.type = 'TemporalCorrelationSampler';


    config.illumination2 = config.illumination;
    config.view2 = config.view;

    config.tracer.isCBS = false;
    tic
    [u, iter] = sstmc2D(config);
    runTimeCorrelation_noCBS = toc

    C_correlation_noCBS = u / iter;

    config.tracer.isCBS = true;
    tic
    [u, iter] = sstmc2D(config);
    runTimeCorrelation_CBS = toc

    C_correlation_CBS = u / iter;
end

%% Run field
if(fieldAlgorithmValidation)

    config.simulation.fullIteration = 1;
    config.simulation.renderingsNum = 1000;
    config.sampler.type = 'TemporalSampler';

    config.tracer.isCBS = false;
    tic
    [u, iter] = sstmc2D(config);
    runTimeField = toc

    C_field_noCBS = mean(u .* conj(u) ./ permute(iter,[1,3,4,2]),4);

    config.tracer.isCBS = true;
    tic
    [u, iter] = sstmc2D(config);
    runTimeField = toc

    C_field_CBS = mean(u .* conj(u) ./ permute(iter,[1,3,4,2]),4);
end

%% Plot
figure

if(correlationAlgorithmValidation)
    subplot(1,2,1)
    hold on

    plot(rad2deg(mudiffData.degList), abs(mudiffData.mudiffRes));
    plot(rad2deg(mudiffData.degList), abs(C_correlation_noCBS));
    plot(rad2deg(mudiffData.degList), abs(C_correlation_CBS));

    legend('mudiff','MC correlation no CBS','MC correlation CBS')

    xlabel('time (s)')
    ylabel('speckle intensity')
    xlim([0,360])

    title('Mudiff v.s. correlation MC 2D')
end

if(fieldAlgorithmValidation)
    subplot(1,2,2)
    hold on

    plot(rad2deg(mudiffData.degList), abs(mudiffData.mudiffRes));
    plot(rad2deg(mudiffData.degList), abs(C_field_noCBS));
    plot(rad2deg(mudiffData.degList), abs(C_field_CBS));

    legend('mudiff','MC field no CBS','MC field CBS')

    xlabel('time (s)')
    ylabel('speckle intensity')
    xlim([0,360])

    title('Mudiff v.s. field MC 2D')
end

