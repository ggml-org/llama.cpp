fig1 = figure(1);
clf;

%orient(fig1,'landscape')
orient(gcf,'landscape')

for i = 1:128
    subplot(8,16,i);
    xx = 400:400:2000;
    aa = rand(size(xx));
    plot(xx,aa);
end

% broken.
if 0
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [60 36]);
set(fig1,'PaperUnits','normalized');
set(fig1,'PaperPosition', [0 0 1 1]);
print(fig1, 'testrand', '-dpdf');
end

if 0
% works okay.
set(gcf,'PaperUnits', 'inches');
set(gcf,'PaperSize', [72 36]);
set(gcf,'PaperPositionMode','auto');         
set(gcf,'PaperOrientation','landscape');
set(gcf,'Position',[50 50 4000 1800]);
print(gcf, 'testrand','-bestfit','-dpdf');
end

if 1
% works better?
set(gcf,'Position',[0 0 2000 900]);
set(gcf,'PaperUnits', 'inches');
set(gcf,'PaperSize', [48 22]);
set(gcf,'PaperPosition', [0 0 48 22]);
%set(gcf,'PaperPositionMode','auto');         
set(gcf,'PaperPositionMode','manual');         
set(gcf,'PaperOrientation','landscape');
print(gcf, 'testrand','-bestfit','-dpdf');
end

