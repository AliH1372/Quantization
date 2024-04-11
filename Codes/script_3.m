%% Defining Parameters 

modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
  "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
  "B-FM", "DSB-AM", "SSB-AM"]);


% Set the random number generator to a known state to be able to regenerate
% the same frames every time the simulation is run
rng(123456)
% Random bits
d = randi([0 3], 1024, 1);
% PAM4 modulation
syms = pammod(d,4);
% Square-root raised cosine filter
filterCoeffs = rcosdesign(0.35,4,8);
tx = filter(filterCoeffs,1,upsample(syms,8));

% Channel
SNR = 30;
maxOffset = 5;
fc = 902e6;
fs = 200e3;
multipathChannel = comm.RicianChannel(...
  'SampleRate', fs, ...
  'PathDelays', [0 1.8 3.4] / 200e3, ...
  'AveragePathGains', [0 -2 -10], ...
  'KFactor', 4, ...
  'MaximumDopplerShift', 4);

frequencyShifter = comm.PhaseFrequencyOffset(...
  'SampleRate', fs);

% Apply an independent multipath channel
reset(multipathChannel)
outMultipathChan = multipathChannel(tx);

% Determine clock offset factor
clockOffset = (rand() * 2*maxOffset) - maxOffset;
C = 1 + clockOffset / 1e6;

% Add frequency offset
frequencyShifter.FrequencyOffset = -(C-1)*fc;
outFreqShifter = frequencyShifter(outMultipathChan);

% Add sampling time drift
t = (0:length(tx)-1)' / fs;
newFs = fs * C;
tp = (0:length(tx)-1)' / newFs;
outTimeDrift = interp1(t, outFreqShifter, tp);

% Add noise
rx = awgn(outTimeDrift,SNR,0);
%% Dataset Generation Parameters

numFramesPerModType = 10000;

sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
symbolsPerFrame = spf / sps;
fs = 200e3;             % Sample rate
fc = [902e6 100e6];     % Center frequencies


%% Create Channel Impairments

maxDeltaOff = 5;
deltaOff = (rand()*2*maxDeltaOff) - maxDeltaOff;
C = 1 + (deltaOff/1e6);

channel = helperModClassTestChannel(...
  'SampleRate', fs, ...
  'SNR', SNR, ...
  'PathDelays', [0 1.8 3.4] / fs, ...
  'AveragePathGains', [0 -2 -10], ...
  'KFactor', 4, ...
  'MaximumDopplerShift', 4, ...
  'MaximumClockOffset', 5, ...
  'CenterFrequency', 902e6);

chInfo = info(channel);
  %% Waveform Generation

% Set the random number generator to a known state to be able to regenerate
% the same frames every time the simulation is run
rng(1235)
tic

numModulationTypes = length(modulationTypes);

channelInfo = info(channel);
transDelay = 50;

%availableGPUs = cpuDeviceCount("available");
pool = parpool('Processes');

if ~isa(pool,"parallel.ClusterPool")
  dataDirectory = fullfile('C:\Users\alimohammad.hakimi\Desktop\Matlabdata3',"ModClassDataFiles");
else
  dataDirectory = uigetdir("","Select network location to save data files");
end
disp("Data file directory is " + dataDirectory)

%%
fileNameRoot = "frame";

% Check if data files exist
dataFilesExist = false;
if exist(dataDirectory,'dir')
  files = dir(fullfile(dataDirectory,sprintf("%s*",fileNameRoot)));
  if length(files) == numModulationTypes*numFramesPerModType
    dataFilesExist = true;
  end
end

if ~dataFilesExist
  disp("Generating data and saving in data files...")
  [success,msg,msgID] = mkdir(dataDirectory);
  if ~success
    error(msgID,msg)
  end
  for modType = 1:numModulationTypes
    elapsedTime = seconds(toc);
    elapsedTime.Format = 'hh:mm:ss';
    fprintf('%s - Generating %s frames\n', ...
      elapsedTime, modulationTypes(modType))
    
    label = modulationTypes(modType);
    numSymbols = (numFramesPerModType / sps);
    dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
    modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
    if contains(char(modulationTypes(modType)), {'B-FM','DSB-AM','SSB-AM'})
      % Analog modulation types use a center frequency of 100 MHz
      channel.CenterFrequency = 100e6;
    else
      % Digital modulation types use a center frequency of 902 MHz
      channel.CenterFrequency = 902e6;
    end
    
    for p=1:numFramesPerModType
      % Generate random data
      x = dataSrc();
      
      % Modulate
      y = modulator(x);
      
      % Pass through independent channels
      rxSamples = channel(y);
      
      % Remove transients from the beginning, trim to size, and normalize
      frame = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
      
      % Save data file
      fileName = fullfile(dataDirectory,...
        sprintf("%s%s%03d",fileNameRoot,modulationTypes(modType),p));
      save(fileName,"frame","label")
    end
  end
else
  disp("Data files exist. Skip data generation.")
end

%% Create a Datastore

frameDS = signalDatastore(dataDirectory,'SignalVariableNames',["frame","label"]);

%% Transform Complex Signals to Real Arrays

frameDSTrans = transform(frameDS,@helperModClassIQAsPages);

%% Split into Training, Validation, and Test

percentTrainingSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;

splitPercentages = [percentTrainingSamples,percentValidationSamples,percentTestSamples];
[trainDSTrans,validDSTrans,testDSTrans] = helperModClassSplitData(frameDSTrans,splitPercentages);

%% Import Data into Memory

%Read the training and validation frames into the memory
pctExists = parallelComputingLicenseExists();
trainFrames = transform(trainDSTrans, @helperModClassReadFrame);
rxTrainFrames = readall(trainFrames,"UseParallel",pctExists);
rxTrainFrames = cat(4, rxTrainFrames{:});
validFrames = transform(validDSTrans, @helperModClassReadFrame);
rxValidFrames = readall(validFrames,"UseParallel",pctExists);
rxValidFrames = cat(4, rxValidFrames{:});

% Read the training and validation labels into the memory
trainLabels = transform(trainDSTrans, @helperModClassReadLabel);
rxTrainLabels = readall(trainLabels,"UseParallel",pctExists);
validLabels = transform(validDSTrans, @helperModClassReadLabel);
rxValidLabels = readall(validLabels,"UseParallel",pctExists);

%% DL Model
 
numModTypes = numel(modulationTypes);
modClassNet = [ ...
        imageInputLayer([1 1024 2],'Name','Input Layer','Normalization','none')

        convolution2dLayer([1 4],16,'Name','CNN1','Stride',[1 1],'Padding','same','NumChannels',2)
        batchNormalizationLayer("Name","BN1")
        reluLayer("Name",'ReLU1')
        maxPooling2dLayer([1 2],'Name','MaxPool1','Stride',[1 2])

        convolution2dLayer([1 2],24,'Name','CNN2','Stride',[1 1],'Padding','same','NumChannels',16)
        batchNormalizationLayer("Name","BN2")
        reluLayer("Name",'ReLU2')
        maxPooling2dLayer([1 2],'Name','MaxPool2','Stride',[1 2])

        convolution2dLayer([1 2],32,'Name','CNN3','Stride',[1 1],'Padding','same','NumChannels',24)
        batchNormalizationLayer("Name","BN3")
        reluLayer("Name",'ReLU3')
        maxPooling2dLayer([1 2],'Name','MaxPool3','stride',[1 2])

        convolution2dLayer([1 2],48,'Name','CNN4','Stride',[1 1],'Padding','same','NumChannels',32)
        batchNormalizationLayer("Name","BN4")
        reluLayer("Name",'ReLU4')
        maxPooling2dLayer([1 2],'Name','MaxPool4','stride',[1 2])

        convolution2dLayer([1 2],64,'Name','CNN5','Stride',[1 1],'Padding','same','NumChannels',48)
        batchNormalizationLayer("Name","BN5")
        reluLayer("Name",'ReLU5')
        maxPooling2dLayer([1 2],'Name','MaxPool5','Stride',[1 2])

        convolution2dLayer([1 2],96,'Name','CNN6','Stride',[1 1],'Padding','same','NumChannels',64)
        batchNormalizationLayer("Name","BN6")
        reluLayer("Name",'ReLU6')
        maxPooling2dLayer([1 2],'Name','MaxPool6','Stride',[1 2])

        convolution2dLayer([1 2],128,'Name','CNN7','Stride',[1 1],'Padding','same','NumChannels',96)
        batchNormalizationLayer("Name","BN7")
        reluLayer("Name",'ReLU7')
        averagePooling2dLayer([1 16],"Name",'AP1','Stride',[1 1])

        fullyConnectedLayer(11,'Name','Fully Connected')
        softmaxLayer("Name",'softmax')
        classificationLayer("Name",'Output')

    ];

%% Train the CNN

maxEpochs = 20;
miniBatchSize = 1024;

options = helperModClassTrainingOptions(maxEpochs,miniBatchSize,...
  numel(rxTrainLabels),rxValidFrames,rxValidLabels);


 trainedNet = trainNetwork(rxTrainFrames,rxTrainLabels,modClassNet,options);


%% Accuracy Measurments

% Read the test frames into the memory
testFrames = transform(testDSTrans, @helperModClassReadFrame);
rxTestFrames = readall(testFrames,"UseParallel",pctExists);
rxTestFrames = cat(4, rxTestFrames{:});

% Read the test labels into the memory
testLabels = transform(testDSTrans, @helperModClassReadLabel);
rxTestLabels = readall(testLabels,"UseParallel",pctExists);

rxTestPred = classify(trainedNet,rxTestFrames);
testAccuracy = mean(rxTestPred == rxTestLabels);
disp("Test accuracy: " + testAccuracy*100 + "%")

%% Confusion Matrix

figure
cm = confusionchart(rxTestLabels, rxTestPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.Parent.Position = [cm.Parent.Position(1:2) 950 550];

%% Estimate network metrics

%net = SeriesNetwork(modClassNet);


%estNet = estimateNetworkMetrics(trainedNet)


%% Augmentation for calibration


aug_calData = augmentedImageDatastore([1,1024,2], rxValidFrames, rxValidLabels);
aug_valdata = augmentedImageDatastore([1,1024,2], rxTestFrames, rxTestLabels);

%% Quatization object

quantObj = dlquantizer(trainedNet,'ExecutionEnvironment','MATLAB');

%% Calibration

calResults = calibrate(quantObj,aug_calData);

%% Quantize the network

qNet = quantize(quantObj);

%% Model Parameters

qDetailsQuantized = quantizationDetails(qNet);

%% Accuracy Measurement comparison 

predOriginal = classify(trainedNet,aug_valdata);       
predQuantized = classify(qNet,aug_valdata);      

Quantized = mean(predQuantized == rxTestLabels)*100

Original = mean(predOriginal == rxTestLabels)*100
%% Model Parameters Size comparison
estNet_1 = estimateNetworkMetrics(qNet)
estNet_2 = estimateNetworkMetrics(trainedNet)





















