[rows, cols, dimen1, dimen2] = size(video); %if it doesn't work, remove the commas
for j = 31751:dimen2
image = video(:,:,:,j);
imwrite(image, strcat('image',int2str(j),'.png'));
end