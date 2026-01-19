import winsound
import threading
import time
import random


class IntenseSirenSound:
    def __init__(self):
        self.running = False

    def extreme_loud_siren(self, duration=30):
        """
        Create an extremely loud and chaotic siren sound
        """
        self.running = True
        start_time = time.time()

        def intense_siren_pattern():
            while self.running and time.time() - start_time < duration:
                # Randomized, aggressive frequency shifts
                for _ in range(20):  # Multiple rapid sound bursts
                    if not self.running:
                        break

                    # Extremely rapid and random frequency changes
                    freq1 = random.randint(500, 2000)
                    freq2 = random.randint(1000, 3000)

                    # Loud, short bursts with unpredictable frequencies
                    winsound.Beep(freq1, 50)  # Very short, sharp sound
                    winsound.Beep(freq2, 70)  # Slightly longer, different pitch

                # Occasional super loud, sustained tone
                winsound.Beep(2500, 300)  # Piercing, sustained high-pitch

                # Brief pause to create rhythmic effect
                time.sleep(0.1)

        # Run in a separate thread to prevent blocking
        siren_thread = threading.Thread(target=intense_siren_pattern)
        siren_thread.start()
        return siren_thread

    def nuclear_alarm_siren(self, duration=30):
        """
        Ultra-aggressive, panic-inducing siren sound
        """
        self.running = True
        start_time = time.time()

        def nuclear_siren_pattern():
            while self.running and time.time() - start_time < duration:
                # Extremely aggressive sound pattern
                for _ in range(15):
                    if not self.running:
                        break

                    # Rapid, jarring frequency shifts
                    winsound.Beep(random.randint(1500, 3000), 30)
                    winsound.Beep(random.randint(500, 1200), 40)

                # Ear-piercing sustained tones
                winsound.Beep(3000, 200)  # Extremely high pitch
                winsound.Beep(400, 150)  # Deep, rumbling tone

                # Short pause to create tension
                time.sleep(0.05)

        # Run in a separate thread
        siren_thread = threading.Thread(target=nuclear_siren_pattern)
        siren_thread.start()
        return siren_thread

    def stop_siren(self):
        """
        Immediately stop the siren
        """
        self.running = False


def main():
    # Create intense siren object
    loud_siren = IntenseSirenSound()

    print("EXTREMELY LOUD SIREN ACTIVATED!")

    # Choose your preferred intense siren mode

    # Option 1: Extreme Loud Siren
    siren_thread = loud_siren.extreme_loud_siren(duration=15)

    # Alternatively, try the nuclear alarm
    # siren_thread = loud_siren.nuclear_alarm_siren(duration=15)

    try:
        # Keep main thread running
        siren_thread.join()
    except KeyboardInterrupt:
        # Allow stopping with Ctrl+C
        loud_siren.stop_siren()

    print("Siren Terminated!")


if __name__ == "__main__":
    main()