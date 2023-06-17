%% reconeixem les imatges de test usant la xarxa 

  clear all
  close all
%  



xarxa = load('network.mat');

imagefiles = dir('./network/test_images/Alladin/*.jpg');      
nfiles = length(imagefiles);    % Number of files found


for i = 1:nfiles
    filename = ['./network/test_images/Alladin/' imagefiles(i).name];
    im = imread(filename);
    
    im=imresize(im,[227 227]);
    %[YPred,probs] = classify(trainedNetwork_1,im);
    %imshow(im)
    %title(string(YPred) + ", " + num2str(100*max(probs),3) + "%");
    %i

    [bbox, score, label] = detect(xarxa, im, 'MiniBatchSize', 32);

    [score, idx] = max(score);

    bbox = bbox(idx, :);
    annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

    detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

    figure
    imshow(detectedImg)
    pause
end
disp('se acabo')


