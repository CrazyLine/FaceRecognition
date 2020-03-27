clear;close all;clc

downheight=40;
downwidth=60;
num=20;
testnum=20;

imds = imageDatastore('Training', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

figure;
Data=[];
for i=1:numel(imds.Labels)
    temp = readimage(imds,i);
    temp=rgb2gray(imresize(temp,[downheight, downwidth]));
    subplot(5 , 8 , i);
    imshow(temp);
    title(strcat("RNo.",num2str(i)));
    hold on;
    temp = im2double(temp);
    Data= [Data reshape(temp, [], 1)];   %convert one image data into a column
end


for i=1:num
    path = strcat('Training2/','R_img',num2str(i),'.jpg');
    temp = imread(string(path));
    temp=rgb2gray(imresize(temp,[downheight, downwidth]));
    subplot(5 , 8 , i+20);
    imshow(temp);
    title(strcat("RNo.",num2str(i+20)));
    hold on;
    temp = im2double(temp);
    Data= [Data reshape(temp, [], 1)];   %convert one image data into a column
end

MeanValue = mean(Data, 2);
OriginalMean = repmat(MeanValue,1, size(Data,2));
DataAdjust = Data - OriginalMean;

figure;
meanimage=reshape(MeanValue,[downheight,downwidth]);
imshow(meanimage);
title("Mean Image");

[row, col] = size(Data);
if row >= col
    length = col;
else
    length = row;
end

% first method
Covariance = cov(DataAdjust');
[EigenVectors, EigenValues] = eig(Covariance); %The return value eigenvalues is sorted from small to large, so select the principal component from the next column
SortEigenVectors = fliplr(EigenVectors); %Feature vector sorting: the first column has the largest corresponding feature value, followed by the next one
FeatureVectors = SortEigenVectors(:,1:length-1); %select part the EigenVectors as the components, here is the first length-1 columns

%Select a feature vector to display, that is, the so-called feature face. Currently, there are 40-1=39 feature faces
figure;
for i=1:length-1
EigenFace = reshape(FeatureVectors(:,i),downheight, downwidth);
subplot(8 , 5 , i );
imshow(EigenFace,[]);
titlename=strcat("ENo.",num2str(i));
title(titlename);
hold on;
end


%Image data projection to feature space
RowFeatureVector  = FeatureVectors';
PatternVectors = DataAdjust' * RowFeatureVector'; %Each row represents an image projected into n-1 dimensional space.

% load Testing
imds1 = imageDatastore('Testing', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% 9 images from Testing
index1=[2,3,8,9,11,14,15,18,19];
% 8 images from Testing2 headshot
index2=[1,2,3,4,5,6,7,8];
% 3 images from Testing2 non-headshot
index3=[1,3,5];

TPVs=[];
myindexes=[];
figure;
for i=1:testnum

if i<=9
    TestData = readimage(imds1,index1(i));
elseif i>9 && i<=17
    path = strcat('Testing2/','T_imgo',num2str(index2(i-9)),'.jpg');
    TestData = imread(string(path));
else
    path = strcat('Testing2/','TT_imgo',num2str(index3(i-17)),'.jpg');
    TestData = imread(string(path));
end

temp=rgb2gray(imresize(TestData,[downheight, downwidth]));
subplot(5 , 4 , i);
imshow(temp);
title(strcat('TNo.',num2str(i)));
hold on;
temp = im2double(temp);%*illumination_factor;
TestData = reshape(temp, [], 1);
TestDataPatternVector = (TestData - MeanValue)' * FeatureVectors; 
TPVs=[TPVs;TestDataPatternVector];
%Euclidean distance is used for comparison.
result(length,1) = 0;
for j=1:length
    temp = (PatternVectors(j,:) - TestDataPatternVector).^2;
    result(j,1) = sqrt(sum(temp));
end
[minimum, index] = min(result);
X = strcat("TNo.",num2str(i)," match the ",'RNo.',num2str(index));
myindexes=[myindexes,index];
disp(string(X));
myloss=strcat('loss: ',num2str(minimum));
disp(myloss);

end

figure;
pos=6;
for i=1:numel(myindexes)
    if myindexes(i)<=20
        TrainingData= readimage(imds,myindexes(i));
    else
        path = strcat('Training2/','R_img',num2str(myindexes(i)-20),'.jpg');
        TrainingData = imread(string(path));
    end
    temp=rgb2gray(imresize(TrainingData,[downheight, downwidth]));
    subplot(8 , 5 , pos);
    if mod(i,5)==0
        pos=pos+6;
    else
        pos=pos+1;
    end
    imshow(temp);
    title(strcat("MNo.",string(i))); % mached no
    hold on;
end

pos1=1;
for i=1:testnum

if i<=9
    TestData = readimage(imds1,index1(i));
elseif i>9 && i<=17
    path = strcat('Testing2/','T_imgo',num2str(index2(i-9)),'.jpg');
    TestData = imread(string(path));
else
    path = strcat('Testing2/','TT_imgo',num2str(index3(i-17)),'.jpg');
    TestData = imread(string(path));
end

temp=rgb2gray(imresize(TestData,[downheight, downwidth]));
subplot(8 , 5 , pos1);
if mod(i,5)==0
    pos1=pos1+6;
else
    pos1=pos1+1;
end
imshow(temp);
title(strcat('TNo.',num2str(i)));
hold on;

end

% two coefficients
trainingcoefficient=PatternVectors';% 39 x 40
TestDataPatternVectors=TPVs; % 20x39


