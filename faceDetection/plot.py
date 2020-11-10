import matplotlib.pyplot as plt
import numpy as np
# wk_train = [87.71428571428572, 87.39285714285714, 87.42857142857143, 87.64285714285714, 86.95, 88.6, 87.65, 87.79999999999998, 87.66666666666666, 88.83333333333333, 89.0, 86.66666666666669, 88.00000000000001, 87.25, 88.00000000000001, 88.75]
# wk_test = [87.44769874476987, 88.11715481171548, 88.20083682008368, 86.52719665271967, 86.81704260651628, 85.91478696741855, 86.61654135338345, 86.11528822055136, 86.86940966010732, 86.22540250447226, 86.33273703041144, 87.0840787119857, 86.73157162726008, 83.25452016689846, 85.64673157162727, 85.5076495132128]
# boost_train = [87.7142857142857, 87.39285714285714, 87.50000000000002, 87.64285714285714, 86.950, 88.600, 87.900, 87.350, 87.66666666666666, 88.83333333333334, 89.000, 87.33333333333332, 88.000, 87.750, 89.250, 90.000]
# boost_test = [87.44769874476987, 88.11715481171548, 88.11715481171548, 86.52719665271967, 86.81704260651628, 85.91478696741855, 86.8671679197995, 86.8671679197995, 86.86940966010732, 85.97495527728086, 86.33273703041144, 87.871198568873, 86.73157162726008, 83.5326842837274, 85.8414464534075, 86.56467315716272]


# wk_train = [87.58823529411764, 87.87777777777778, 87.78333333333333, 87.83333333333333, 89.0]
# wk_test = [86.10644257703082, 86.03015075376884, 87.08437761069341, 86.45520311630493, 85.92047128129603]
# boost_train = [87.65686274509802, 87.87777777777778, 88.23333333333332, 88.16666666666666, 89.05555555555556]
# boost_test = [86.21848739495799, 86.03015075376884, 87.468671679198, 86.72231496939344, 86.03829160530192]
#
# boost_train_12 = [87.51960784313727, 87.51111111111112, 87.46666666666665, 88.10000000000001, 89.05555555555556]
# boost_test_12 = [86.89075630252103, 87.4706867671692, 86.98412698412697, 86.9449081803005, 86.70594010800198]

matrix = np.array([[87.46078431, 87.61764706, 87.44117647, 86.49859944, 86.77871148, 86.55462185],
          [87.43333333, 87.54444444, 87.42222222, 88.14070352, 88.10720268, 88.00670017],
[87.81666667, 88.2,        87.71666667, 86.66666667, 87.06766917, 86.61654135],
[88.56666667, 88.66666667, 88.76666667, 86.51085142, 86.5442404,  86.74457429],
[88.83333333, 89.38888889, 88.  ,       86.42120766, 86.40157094, 84.97790869]])

# [2, 4, 8, 12]:
# x = [0.3, 0.5, 0.7, 0.9]
x = [0.15, 0.25, 0.5, 0.75, 0.85]
plt.plot(x, matrix[:,0], color='blue', label="wk_train")
plt.plot(x, matrix[:,3], color='blue', label="wk_test", linestyle='dashed')
plt.plot(x, matrix[:,1], color='green', label="boost_train_4")
plt.plot(x, matrix[:,4], color='green', label="boost_test_4", linestyle='dashed')
plt.plot(x, matrix[:,2], color='purple', label="boost_train_12")
plt.plot(x, matrix[:,5], color='purple', label="boost_test_12", linestyle='dashed')
plt.ylabel('Accuracy')
plt.xlabel('Training/Test Proportion')
plt.legend()
plt.title("Faces 94 Classification Results")
plt.savefig("plot_2.png")
#
#
# plt.plot(x, wk_train[4:8], color='blue', label="wk_train")
# plt.plot(x, wk_test[4:8], color='red', label="wk_test")
# plt.plot(x, boost_train[4:8], color='green', label="boost_train")
# plt.plot(x, boost_test[4:8], color='orange', label="boost_test")
# plt.ylabel('Accuracy')
# plt.xlabel('Training/Test Proportion')
# plt.legend()
# plt.title("Boosting Iterations = 4")
# plt.savefig("plot_4.png")
#
#
# plt.plot(x, wk_train[8:12], color='blue', label="wk_train")
# plt.plot(x, wk_test[8:12], color='red', label="wk_test")
# plt.plot(x, boost_train[8:12], color='green', label="boost_train")
# plt.plot(x, boost_test[8:12], color='orange', label="boost_test")
# plt.ylabel('Accuracy')
# plt.xlabel('Training/Test Proportion')
# plt.legend()
# plt.title("Boosting Iterations = 8")
# plt.savefig("plot_8.png")


# plt.plot(x, wk_train[12:16], color='blue', label="wk_train")
# plt.plot(x, wk_test[12:16], color='red', label="wk_test")
# plt.plot(x, boost_train[12:16], color='green', label="boost_train")
# plt.plot(x, boost_test[12:16], color='orange', label="boost_test")
# plt.ylabel('Accuracy')
# plt.xlabel('Training/Test Proportion')
# plt.legend()
# plt.title("Boosting Iterations = 12")
# plt.savefig("plot_12.png")