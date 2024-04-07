import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from scipy import signal

#definicje funkcji wejsciowych
def step_input(t):
    return 1 if t >= 0 else 0

def sinusoidal_input(t, omega=1):
    return np.sin(omega * t)

# równania stanu systemu
def state_equations(x, t, A, B, u_t):
    return A @ x + B.flatten() * u_t  # zeby zwrócić jednowymiarowy wektor; u_t wartosc skalarna w czasie t
    return x_dot
# symulacja systemu
def simulate_system(A, B, C, D, input_func, t_end, dt=0.01, omega=1):
    t = np.arange(0, t_end, dt)
    u = np.array([input_func(ti, omega) if input_func == sinusoidal_input else input_func(ti) for ti in t])
    x0 = np.zeros(A.shape[0])
    
    #  skalarna wartość do state_equations
    sol = odeint(lambda x, t: state_equations(x, t, A, B, u[min(int(t/dt), len(u)-1)]), x0, t)
    
    y = np.array([C.dot(sol[i]) + D * u[i] for i in range(len(t))])  # D jest skalarne
    return t, u, y



# sprawdzenie stabilnosci
def check_stability(A):
    eigenvalues = np.linalg.eigvals(A)
    print("Wartości własne systemu:", eigenvalues)
    unstable = any(e.real >= 0 for e in eigenvalues)
    if unstable:
        print("System jest niestabilny.")
        return False
    else:
        print("System jest stabilny.")
        return True


# wprowadzanie parametrow od uzytkownika
def get_user_input():
    print("Wprowadź wartości współczynników systemu:")
    a0 = float(input("a0 = "))
    a1 = float(input("a1 = "))
    a2 = float(input("a2 = "))
    b0 = float(input("b0 = "))
    b1 = float(input("b1 = "))
    b2 = float(input("b2 = "))
    b3 = float(input("b3 = "))
    
    # definicja A, B, C, D
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [-a0, -a1, -a2]])
    B = np.array([[0], [0], [1]]) # wektor kolumnowy#jednak potem
    C = np.array([b0 - a0*b3, b1 - a1*b3, b2 - a2*b3])
    D = b3

    return a0, a1, a2, b0, b1, b2, b3



# main
if __name__ == "__main__":
    a0, a1, a2, b0, b1, b2, b3 = get_user_input()

    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [-a0, -a1, -a2]])
    B = np.array([[0], [0], [1]])  
    C = np.array([b0-a0*b3,b1-a1*b3,b2-a2*b3])  
    D = b3  

    #obiekt lokalny
    system=signal.lti(A,B,C,D)  
    # grafy bodego
    frequencies = np.logspace(-2, 2, 1000)  # zakres f
    w, mag, phase = signal.bode(system, frequencies)

    plt.figure()
    plt.semilogx(w, mag)  # wykres bodego
    plt.title('Charakterystyka amplitudowa Bodego')
    plt.xlabel('Częstotliwość [rad/s]')
    plt.ylabel('Amplituda [dB]')
    plt.grid(which='both', linestyle='--')

    plt.figure()
    plt.semilogx(w, phase)  # wykres fazowy bodego
    plt.title('Charakterystyka fazowa Bodego')
    plt.xlabel('Częstotliwość [rad/s]')
    plt.ylabel('Faza [stopnie]')
    plt.grid(which='both', linestyle='--')

    plt.show()

    if check_stability(A):
        input_choice = input("Wybierz pobudzenie (1: skok, 2: sinusoida): ")
        if input_choice == "1":
            input_func = step_input
        elif input_choice == "2":
            input_func = sinusoidal_input
        else:
            print("Nieprawidłowy wybór. Używam skoku jednostkowego.")
            input_func = step_input

        t_end = 10
        dt = 0.01
        omega = 1  # tylko dla sinusoidy
        t, u, y = simulate_system(A, B, C, D, input_func, t_end, dt, omega)

        plt.plot(t, y, label='Odpowiedź systemu')
        plt.plot(t, u, 'r--', label='Pobudzenie')
        plt.title('Odpowiedź systemu na pobudzenie')
        plt.xlabel('Czas')
        plt.ylabel('Wyjście systemu')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Analiza stabilności wykazała, że system jest niestabilny. Proszę sprawdzić wprowadzone parametry.")



