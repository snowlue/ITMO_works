clc;
clear all;

A=[-5 -2; 1 -2];
A1=[2 1; 0 -1];
n=2;            

P=sdpvar(n);   
Q=sdpvar(n);
TH=blkvar;       
TH(1,1)=A'*P+P*A+Q;
TH(1,2)=P*A1;
TH(2,2)=-Q;
TH=sdpvar(TH);


F=[TH<=-0.0001,P>=0.0001,Q>=0.0001];
%Задаем настройки
options=sdpsettings('solver','sedumi','verbose',0);
%Решаем задачу
sol=optimize(F)
%В командном окне получим информацию о том, разрешимы ли заданные матричные
%неравенста. В данном случае имеем: "info: 'Successfully solved
%(SeDuMi-1.3)'". Значит система матричных неравенств разрешима, и исходная
%система устойчива.
P_value = value(P);
Q_value = value(Q);

disp('Matrix P:');
disp(P_value);

disp('Matrix Q:');
disp(Q_value);

 