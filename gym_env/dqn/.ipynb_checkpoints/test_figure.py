import matplotlib.pyplot as plt

r0 = {0: 62.0, 1600: 154.2, 900: 114.8, 200: 9.3, 1900: 165.3, 1100: 118.2, 400: 9.4, 2200: 193.5, 1300: 110.7, 600: 27.2, 1700: 154.3, 1500: 136.5, 2400: 200.0, 800: 121.3, 2000: 170.9, 100: 9.2, 1000: 104.6, 2300: 194.6, 300: 9.2, 1200: 102.2, 1800: 164.7, 500: 11.0, 1400: 123.1, 2100: 164.9, 700: 157.6}

r1 = {0: 9.3, 1600: 105.9, 2700: 179.2, 900: 106.2, 2600: 167.1, 200: 9.5, 1900: 133.6, 1100: 138.7, 400: 9.5, 2200: 141.2, 1300: 97.4, 600: 40.5, 1700: 128.9, 2900: 177.4, 1500: 133.7, 2400: 113.7, 800: 88.8, 2000: 126.5, 100: 9.2, 1000: 128.6, 2300: 155.2, 3200: 179.9, 300: 9.1, 3100: 182.2, 1200: 118.1, 1800: 125.9, 3000: 184.1, 500: 26.6, 1400: 118.5, 2100: 138.1, 700: 77.2, 2800: 184.1, 2500: 127.0}

x0 = [key for key in r0]
x0 = sorted(x0)
y0 = [r0[key] for key in x0]

x = [key for key in r1]
x = sorted(x)
y = [r1[key] for key in x]

plt.figure(figsize=(6, 5))
plt.subplot(111)
plt.xlabel('Steps')
plt.ylabel('Avg rewards of one episode')
plt.xlim(0, 10000)
# plt.ylim(0, max(y))
plt.ylim(0, 300)
plt.plot(x0, y0, label='average rewards of 10 test episode(DQN1)', color='red', linewidth=1)
plt.plot(x, y, label='average rewards of 10 test episode(DQN2)', color='green', linewidth=1)
plt.legend(loc='lower right')
plt.show()