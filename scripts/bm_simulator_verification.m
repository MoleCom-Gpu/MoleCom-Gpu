clear all;
DT = 5e-4;
filecount = 100;
simruntime = 10;
points = 10/DT;
symbolsize = 100000;
global RR;
global R0;
global D;
figure; 
hold on;
%experimental = zeros(points, 6, 'int32');
%actual = zeros(points, 6, 'int32');
for j = 1:6
    sum = zeros(points,1, 'int32');
    for i= 1:filecount
       filename = sprintf('%03d.txt', i);
       foldername = sprintf('results%d', j);
       filename = fullfile('~/Desktop','492_results_1', foldername,filename);
       fileID = fopen(filename);
       C = textscan(fileID, '%d');
       A = cell2mat(C);
       sum = sum + A;
       fclose(fileID);
    end
    switch j
        case 1
            RR = 10;
            R0 = sqrt(260); % 2^2 + 16^2
            D = 79.4;
        case 2
            RR = 10;
            R0 = sqrt(740);
            D = 79.4;
        case 3
            RR = 5;
            R0 = sqrt(305);
            D = 79.4;
        case 4
            RR = 5;
            R0 = sqrt(545);
            D = 79.4;
        case 5
            RR = 10;
            R0 = sqrt(260);
            D = 39.7;
        case 6
            RR = 10;
            R0 = sqrt(740);
            D = 39.7;
        case 7
            RR = 5; %dt:2.5e-4
            R0 = sqrt(305);
            D = 39.7;
    end
    experimental = double(sum) ./ filecount;
    actual = arrayfun(@exprx1, (0:DT:(10-DT))', ones(points,1) .* DT)*symbolsize;
    downsamplingfactor = 5;
    subplot(3, 2, j);
    hold on;
    if mod(j, 2) == 1
        ttl = sprintf('Molecular Communication Simulator Verification: Case %d (FRONT)', j);
         title(ttl);
    else
        ttl = sprintf('Molecular Communication Simulator Verification: Case %d (BEHIND)', j);
        title(ttl);
    end
    
    ylabel('# of received molecules in [t, t+dt]');
    xlabel('Simulation runtime in seconds');
    plot((1:points/downsamplingfactor)/(points/(simruntime*downsamplingfactor)), mean(reshape(experimental, downsamplingfactor, [])),'g*');
    plot((1:points/downsamplingfactor)/(points/(simruntime*downsamplingfactor)),mean(reshape(actual, downsamplingfactor, [])), 'r');
    str = sprintf('RR = %.2f \\mum\nR0 = %.2f \\mum\nD  = %.2f \\mum^2/sec\n', RR, R0, D);
    text(7,max(actual)/3, str); 
    legend('Simulation (Spherical Tx - Spherical Rx)', 'Analytical (Point Tx - Spherical Rx)');
end
function y = exprx1(t, dt)
    y = exprx0(t + dt) - exprx0(t);
end
function y = exprx0(t)
    global RR;
    global R0;
    global D;
    y = (RR/R0)*erfc((R0-RR)/sqrt(4*D*t));
end