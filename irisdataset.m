clc,clear,close all
load iris.dat
x = iris(:,1:4)/10;
y = iris(:,5)';

n=1;

% training set (iris dataset)
figure(1)
for j=1:4
   for i=1:4
        subplot(4,4,n)
        plot(x(1:50,i),x(1:50,j),"r.",x(51:100,i),x(51:100,j),"g.",x(101:150,i),x(101:150,j),"k.")
        hold on
        plot(x(84,i),x(84,j),"b*")
        hold off
        switch n
            case 1
                title('Sepal.Length')
                ylabel('Sepal.Length')
            case 2
                title('Sepal.Width')
            case 3
                title('Petal.Length')
            case 4
                title('Petal.Width')
            case 5
                ylabel('Sepal.Width')
            case 9
                ylabel('Petal.Length')
            case 13
                ylabel('Petal.Width')
        end
        n=n+1;
    end
     
end
sgtitle("Iris data (red=setosa, green=versicolor, black=virginica)")

