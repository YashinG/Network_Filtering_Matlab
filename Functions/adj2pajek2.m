% ADJ2PAJEK2 Converts an adjacency matrix representation to a Pajek .net read format
% adj2pajek2(adj, filename-stem, 'argname1', argval1, ...)
%
% Run pajek (available from http://vlado.fmf.uni-lj.si/pub/networks/pajek/)
% Choose File->Network->Read from the menu
% Then press ctrl-G (Draw->Draw)
% Optional: additionally load the partition file then press ctrl-P (Draw->partition)
%
% Examples
% A=zeros(5,5);A(1,2)=-1;A(2,1)=-1;A(1,[3 4])=1;A(2,5)=1;
% adj2pajek2(A,'foo') % makes foo.net
%
% adj2pajek2(A,'foo','partition',[1 1 2 2 2]) % makes foo.net and foo.clu
%
% adj2pajek2(A,'foo',...
%            'nodeNames',{'TF1','TF2','G1','G2','G3'},...
%            'shapes',{'box','box','ellipse','ellipse','ellipse'});
%
%
% The file format is documented on p68 of the pajek manual
% and good examples are on p58, p72
%
%
% Written by Kevin Murphy, 30 May 2007
% Based on adj2pajek by Gergana Bounova
% http://stuff.mit.edu/people/gerganaa/www/matlab/routines.html
% Fixes a small bug (opens files as 'wt' instead of 'w' so it works in windows)
% Also, simplified her code and added some features.
%
%***************** Re-written by Yashin Gopi to cater for specific process****************

function [] = adj2pajek2(inData, labels, filename, blnColourEdge, EdgeColors)
% Note does not work for N==2

nLabels = size(labels,1) ;
N = size(inData,1) ;

fid = fopen(sprintf('%s.net', filename),'wt','native');
fprintf(fid,'*Vertices %i\n',nLabels);

for i=1:nLabels
    fprintf(fid,'%s\n', labels{i});
end

if size(inData,1) == size(inData,2)

    fprintf(fid,'*Matrix\n'); % directed
    for i=1:N
        for j=1:N
            if j ~=N
                fprintf(fid,'%i ',inData(i,j));
            else
                fprintf(fid,'%i\n',inData(i,j));
            end
        end
    end

else

    fprintf(fid,'*Edges\n');

    for i=1:N

        fprintf(fid,'%i %i ', inData(i,1:2));


        if blnColourEdge == 1
            if exist('EdgeColors','var')
                if EdgeColors(i,1) > 0
                    fprintf(fid,num2str(0.5,'%0.4f'));
                    fprintf(fid,' c Grey\n');

                else
                    fprintf(fid,num2str(2,'%0.4f'));
                    fprintf(fid,' c Maroon\n');

                end
            else
                if inData(i,3) > 0
                    fprintf(fid,num2str(inData(i,3),'%0.4f'));
                    fprintf(fid,' c Black\n');

                else
                    fprintf(fid,num2str(-inData(i,3),'%0.4f'));
                    fprintf(fid,' c Maroon\n');

                end
            end
        else
            fprintf(fid,num2str(inData(i,3),'%0.4f'));
            fprintf(fid,' \n');
            %fprintf(fid,'\n');
            %sprintf('\n')
        end

    end

end
fclose(fid)