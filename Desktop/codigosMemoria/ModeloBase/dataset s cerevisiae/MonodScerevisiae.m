function MonodScerevisiae
close all

%Parámetros y valores iniciales (Kinetics of batch beer fermentation)
umax=0.4;     % h^-1 % tasa máxima de crecimiento específico de la levadura para la fermentación primaria 
umax2=0.2;   %Asumimos que umax no cambia %tasa máxima de crecimiento específico de la levadura para la fermentación secundaria
Yxs=0.47;       %  g/g % coeficiente de rendimiento
Yps=0.43;      % g/g % coeficiente de rendimiento
Ksx=237;        % g/L % constante de saturación
qpmax=1.25;      % tasa específica máxima de acumulación de etanol
Ksp=323;        %g/L %constante de saturación
Pmax=45;        % concentración máxima de etanol a la que se produce la inhibición completa del crecimiento celular y la acumulación de etanol,

%X0=0;          %Biomasa inicial g/L
P0=0;             %Etanol inicial en g/L
S0=100;          %Concentración de sustrato inicial g/L


function dC=fder(t,C)
X=C(1,1);
P=C(2,1); 
S=C(3,1);

u=(umax*S)/(Ksx+S);

dC(1,1)=u;                                                           %dX/dt
dC(2,1)=qpmax*(S/(Ksp+S))*X*(1-P/Pmax);           %dP/dt
dC(3,1)=-(dC(1,1)/Yxs)-(dC(2,1)/Yps);                     %dS/dt

if S<0.19*S0             %Extent 81%
    u=0;
    dC(2,1)=0;
    dC(3,1)=0;
end
if X>97.5                     %[X] máxima
    dC(1,1)=0;
end
end

function dC2=fder2(t,C2)
X=C2(1,1);
P=C2(2,1); 
S=C2(3,1);

u=(umax2*S)/(Ksx+S);

dC2(1,1)=u*X;                                                           %dX/dt
dC2(2,1)=qpmax*(S/(Ksp+S))*X*(1-P/Pmax);               %dP/dt
dC2(3,1)=-(dC2(1,1)/Yxs)-(dC2(2,1)/Yps);                     %dS/dt

if S<0.19*Cf(:,3)             %Extent 81%
    u=0;
        dC2(1,1)=0;
    dC2(2,1)=0;
    dC2(3,1)=0;
end
if X>97.5                     % [X] máxima
    dC2(1,1)=0;
end
end

for X0 = 1.0:1.0:5.0

disp(P0)
 
tode=linspace(0,72,150);

[Tiempo,Conc]=ode23s(@fder,[tode],[X0 P0 S0]);

%Fermentación primaria

Cf = Conc(end, :);


%Fermentación secundaria
tode2=linspace(0,72,150);

[Tiempo,Conc2]=ode23s(@fder2,[tode2],Cf);

Cf2 = Conc2(end, :);
Concatenado = cat(1, Conc, Conc2);
Concatenado



%Gráficos
%Fermentación primaria
%figure
%plot(Tiempo,Conc(:,3),'Color','g','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc(:,1),'Color','[1 0.7 0]','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc(:,2),'Color','[1 0.2 0]','LineStyle','-','LineWidth',2)
%grid on
%title('Fermentación primaria S. cerevisiae')
%legend('Sustrato','Biomasa','Etanol')
%xlabel('Tiempo (h)')
%ylabel('Concentración (g/L)')
%axis([0 72 0 inf])
%text(10,24,'\leftarrow Crecimiento máximo')
%Fermentación secundaria
%figure
%plot(Tiempo,Conc2(:,3),'Color','g','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc2(:,1),'Color','[1 0.7 0]','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc2(:,2),'Color','[1 0.2 0]','LineStyle','-','LineWidth',2)
%grid on
%title('Fermentación secundaria S. cerevisiae')
%legend('Sustrato','Biomasa','Etanol')
%xlabel('Tiempo (h)')
%ylabel('Concentración (g/L)')
%axis([0 50 0 inf])
%Ambas fermentaciones
figure
subplot(1,2,1)
plot(Tiempo,Conc(:,3),'Color','g','LineStyle','-','LineWidth',2)
hold on
plot(Tiempo,Conc(:,1),'Color','[1 0.7 0]','LineStyle','-','LineWidth',2)
hold on
plot(Tiempo,Conc(:,2),'Color','[1 0.2 0]','LineStyle','-','LineWidth',2)
grid on
title('Fermentación primaria S. cerevisiae')
legend('Sustrato','Biomasa','Etanol')
xlabel('Tiempo')
ylabel('Concentración (g/L)')
axis([0 50 0 60])
subplot(1,2,2)
plot(Tiempo,Conc2(:,3),'Color','g','LineStyle','-','LineWidth',2)
hold on
plot(Tiempo,Conc2(:,1),'Color','[1 0.7 0]','LineStyle','-','LineWidth',2)
hold on
plot(Tiempo,Conc2(:,2),'Color','[1 0.2 0]','LineStyle','-','LineWidth',2)
grid on
title('Fermentación secundaria S. Cerevisiae')
legend('Sustrato','Biomasa','Etanol')
xlabel('Tiempo')
ylabel('Concentración (g/L)')
axis([0 50 0 60])
end

%Comparativa
%figure
%yyaxis left
%plot(Tiempo,Conc(:,3),'Color','[0 0.7 0.8]','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc(:,1),'Color','[0 0.7 0.8]','LineStyle','--','LineWidth',2)
%hold on
%plot(Tiempo,Conc(:,2),'Color','[0 0.7 0.8]','LineStyle','-.','LineWidth',2)
%hold on
%ylabel('Concentración fermentación primaria (g/L)')
%yyaxis right
%plot(Tiempo,Conc2(:,3),'Color','[1 0.5 0]','LineStyle','-','LineWidth',2)
%hold on
%plot(Tiempo,Conc2(:,1),'Color','[1 0.5 0]','LineStyle','--','LineWidth',2)
%hold on
%%plot(Tiempo,Conc2(:,2),'Color','[1 0.5 0]','LineStyle','-.','LineWidth',2)
%grid on
%title('Comparativa de las fermentaciones')
%legend('Sustrato','Biomasa','Etanol','Sustrato2','Biomasa2','Etanol2')
%xlabel('Tiempo (h)')
%ylabel('Concentración  fermentación secundaria (g/L)')
%axis([0 50 0 60])


%ConcatenadoT = vertcat(Tiempo, Concatenado)
%ConcatenadoT

end