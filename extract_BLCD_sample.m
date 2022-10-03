
VLFEAT_PATH = 'YOUR_PATH/vlfeat';

%% Setup the pipeline environment.
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
run('YOUR_PATH/matconvnet/matlab/vl_setupnn.m');
run('YOUR_PATH/matconvnet/contrib/autonn/setup_autonn.m')


gpuDevice(0)
load('BLCD(dim256)-Lib.mat','deployNet');
deployNet = Net(deployNet);
deployNet.reset();
deployNet.move('gpu');
     

% Read the image for keypoint detection, patch extraction and
% descriptor computation.
image = imread("YOUR_IMAGE_FILE_PATH");
if ismatrix(image)
    image = single(image);
else
    image = single(rgb2gray(image));
end


%% Compute the keypoints and descriptors.

f = vl_sift(image, 'PeakThresh', 3);
% remove boundary frames
PatchExtend = 7.5;
radius = f(3,:) * PatchExtend;
idx = ((f(1,:) - radius) <= 0 | (f(1,:) + radius) >= size(image,2) | (f(2,:) - radius) <= 0 | (f(2,:) + radius) >= size(image,1));
f(:,idx==1) = [];

if isempty(f)
    keypoints = zeros(0, 4, 'single');
    descriptors = zeros(0, 256, 'single');
    write_keypoints(keypoint_paths{i}, keypoints);
    write_descriptors(descriptor_paths{i}, descriptors);
    continue;
end

keypoints = f';

write_keypoints(keypoint_paths{i}, keypoints);
  
[~, descrs] = vl_covdet(image, 'frames', keypoints', 'descriptor', 'patch', 'PatchRelativeExtent', 7.5, 'PatchResolution', 16, 'PatchRelativeSmoothing', 1) ;


N = size(descrs, 2);
w = sqrt(size(descrs,1)) ;

descriptors = zeros(N,256,'single');

patches32 = reshape(descrs, w, w, 1, []);
patches32 = patches32(1:32,1:32,:,:);

z = reshape(patches32,[],N) ;
z = bsxfun(@minus, z, mean(z,1)) ;
n = std(z,0,1) ;
z = bsxfun(@times, z, 1 ./ (n+1e-12)) ;
patches = reshape(z, 32, 32, 1, []) ;

batchSize = 2048;
for iPatch = 1:batchSize:N
    batchStart = iPatch;
    batchEnd = min(iPatch+batchSize-1, N) ;
    batch = patches32(:, :, :, batchStart : batchEnd);
    batch = gpuArray(batch);
    input = {'data',batch};
    deployNet.eval(input, 'test');
    descriptors(batchStart:batchEnd,:) = single(squeeze(gather(deployNet.getValue('lrn'))) > 0)';
end


% Make sure that each keypoint has one descriptor.
assert(size(keypoints, 1) == size(descriptors, 1));

% Write the descriptors to disk for matching.
write_descriptors("YOUR_DESCRP_FILE_PATH", descriptors);

