%TO DO!!!
% replace this string by the path you saved data set in
sBasePath = 'C:\Users\Michael Aguadze\Documents\GTSRB\Training';
imds = imageDatastore(sBasePath,'IncludeSubfolders',true, ...
'LabelSource','foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7);

%Load Pretrained Network

net = googlenet;
%analyzeNetwork(net)

%Assign input size of the network
inputSize = net.Layers(1).InputSize;

%Replace Final two layers with new layers adapted to the new dataset
if isa(net, 'SeriesNetwork')
   lgraph = layerGraph(net.Layers);
else
   lgraph = layerGraph(net);
end

[learnableLayer, classLayer] = findLayersToReplace(lgraph);
%[learnableLayer, classLayer]

%Number of Classes in new dataset
numClasses = numel(categories(imdsTrain.Labels));

%Replace this fully connected layer with a new fully connected layer with
%the of outpusts equal to the number of classes in the networks
if isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, 'Name', ...
        'new_fc','WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
elseif isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1, numClasses, 'Name', 'new_conv', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);

%Replace classification layer with a new one without labels.
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

%To check that the new layersa are connected correctly, plot the new layer
figure('Units', 'normalized', 'Position', [0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0 10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%Train Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter('RandXReflection', true, ...
    'RandXTranslation', pixelRange,...
    'RandYTranslation', pixelRange,...
    'RandXScale', scaleRange, ...
    'RandYScale', scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs',3, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', valFrequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(augimdsTrain, lgraph, options);

[YPred,probs] = classify(net, augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation, idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

    
