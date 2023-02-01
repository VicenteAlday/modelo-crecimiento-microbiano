function MonodScerevisiae
close all

%Parámetros y valores iniciales (Kinetics of batch beer fermentation)
umax=0.92;     % h^-1 % tasa máxima de crecimiento específico de la levadura para la fermentación primaria 
umax2=0.92;   %Asumimos que umax no cambia %tasa máxima de crecimiento específico de la levadura para la fermentación secundaria
Yxs=0.5;       %  g/g % coeficiente de rendimiento
Yps=0.5;        % g/g % coeficiente de rendimiento
Ksx=98.2;        % ug/L % constante de saturación
qpmax=1.25;      % tasa específica máxima de acumulación de etanol
Ksp=104.3;        %ug/L %constante de saturación
Pmax=45;        % concentración máxima de etanol a la que se produce la inhibición completa del crecimiento celular y la acumulación de etanol,

%X0=0;          %Biomasa inicial g/L
P0=20;             %Etanol inicial en g/L
S0=100;          %Concentración de sustrato inicial g/L

function dC=fder(t,C)
X=C(1,1);
P=C(2,1); 
S=C(3,1);

u=(umax*S)/(Ksx+S); % ecuación 20 del artículo 

dC(1,1)=u;                              %dX/dt
dC(2,1)=qpmax*(S/(Ksp+S))*X*(1-P/Pmax); %dP/dt
dC(3,1)=-(dC(1,1)/Yxs)-(dC(2,1)/Yps);   %dS/dt

if S<0.0*S0             %Extent 100%
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

for X0 = 0.0:1.0:4.0

disp(X0)

tode=linspace(0,300,300);

[Tiempo,Conc]=ode23s(@fder,[tode],[X0 P0 S0]);

%Fermentación primaria

Cf = Conc(end, :);

%Fermentación secundaria
tode2=linspace(0,300,300);

[Tiempo,Conc2]=ode23s(@fder2,[tode2],Cf);




Conc

%Gráficos
%Fermentación primaria
figure
plot(Tiempo,Conc(:,3),'Color','g','LineStyle','-','LineWidth',2)
hold on
plot(Tiempo,Conc(:,1),'Color','[1 0.7 0]','LineStyle','-','LineWidth',2)
hold on
%plot(Tiempo,Conc(:,2),'Color','[1 0.2 0]','LineStyle','-','LineWidth',2)
grid on
title('Curva de crecimiento E. Coli')
legend('Sustrato','Biomasa')
xlabel('Tiempo')
ylabel('Concentración (\mug/L)')
axis([0 72 0 inf])
%text(10,24,'\leftarrow Crecimiento máximo')



end

Cf2 = Conc2(end, :);
end