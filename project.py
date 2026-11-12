import numpy as np
import matplotlib.pyplot as plt

FS = 200
BLOCK_SIZE = 10
WINDOW_SIZE = 256
A = 0.80
DURATION = 60
FREQ_1 = 5
FREQ_2 = 18
AMPLITUDE_2 = 0.35
NOISE_LEVEL = 0.05
N_FREQ = 256

omega = np.linspace(0, np.pi, N_FREQ)
frequency_hz = omega * FS / (2 * np.pi)

x_buffer = np.zeros(WINDOW_SIZE)
write_index = 0
samples_generated = 0
previous_y = 0.0
rng = np.random.default_rng(7)

sample_indices = np.arange(WINDOW_SIZE)
transform_matrix = np.exp(-1j * np.outer(omega, sample_indices))
hann_window = np.hanning(WINDOW_SIZE)
window_sum = max(np.sum(hann_window), 1e-12)
H = 1.0 / (1.0 - A * np.exp(-1j * omega))
plt.close("all")
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.suptitle("Real-Time Z-Transform and Signal Response", fontsize=14)

line_input, = axes[0].plot([], [], label="Input x[n]")
line_output, = axes[1].plot([], [], label="Output y[n]", color="tab:orange")
line_transform, = axes[2].plot(
    frequency_hz, np.zeros(N_FREQ), label="Estimated |X(e^jω)|"
)
axes[2].plot(frequency_hz, np.abs(H), "--", label="Theoretical |H(e^jω)|")

axes[0].set_title("Input Signal")
axes[0].set_ylabel("Amplitude")
axes[1].set_title("System Output")
axes[1].set_ylabel("Amplitude")
axes[2].set_title("Finite-Window Transform and System Frequency Response")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude")

for ax in axes:
    ax.grid(True)
    ax.legend(loc="upper right")
axes[2].set_xlim(0, FS / 2)

time_history = []
input_history = []
output_history = []

try:
    while plt.fignum_exists(fig.number) and samples_generated < FS * DURATION:
        n = np.arange(samples_generated, samples_generated + BLOCK_SIZE)
        t = n / FS
        x_new = (
            np.sin(2 * np.pi * FREQ_1 * t)
            + AMPLITUDE_2 * np.sin(2 * np.pi * FREQ_2 * t)
            + NOISE_LEVEL * rng.standard_normal(BLOCK_SIZE)
        )

        y_new = np.empty(BLOCK_SIZE)
        for k, x_n in enumerate(x_new):
            y_new[k] = x_n + A * previous_y
            previous_y = y_new[k]

        for k in range(BLOCK_SIZE):
            x_buffer[write_index] = x_new[k]
            write_index = (write_index + 1) % WINDOW_SIZE

        samples_generated += BLOCK_SIZE
        time_history.extend(t.tolist())
        input_history.extend(x_new.tolist())
        output_history.extend(y_new.tolist())

        t_plot = np.asarray(time_history[-WINDOW_SIZE:])
        x_plot = np.asarray(input_history[-WINDOW_SIZE:])
        y_plot = np.asarray(output_history[-WINDOW_SIZE:])
        line_input.set_data(t_plot, x_plot)
        line_output.set_data(t_plot, y_plot)

        x_ordered = np.concatenate(
            (x_buffer[write_index:], x_buffer[:write_index])
        )
        x_recent = (x_ordered * hann_window)[::-1]
        Xz = transform_matrix @ x_recent
        line_transform.set_ydata(np.abs(Xz) / window_sum)

        for ax in axes[:2]:
            ax.relim()
            ax.autoscale_view()
        if len(t_plot) > 1:
            axes[0].set_xlim(t_plot[0], t_plot[-1])
            axes[1].set_xlim(t_plot[0], t_plot[-1])

        axes[2].relim()
        axes[2].autoscale_view(scalex=False, scaley=True)
        axes[2].set_xlim(0, FS / 2)

        # Manual GUI update avoids FuncAnimation's deleted-animation warning.
        fig.canvas.draw_idle()
        plt.pause(0.05)

except KeyboardInterrupt:
    print("Simulation stopped by user.")

finally:
    if plt.fignum_exists(fig.number):
        plt.show(block=True)

print("Simulation finished.")
