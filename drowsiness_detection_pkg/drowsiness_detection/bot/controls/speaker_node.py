#!/usr/bin/env python3

"""
ROS2 Node to play audio files.
Subscribes to the /audio_file topic and plays the specified audio file using pygame.
"""
import threading
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pygame



class AudioPlayer:
    """Helper class to play audio files using pygame."""

    def __init__(self):
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"[AudioPlayer] Failed to init mixer (no audio device?): {e}")
        self.lock = threading.Lock()

    def play(self, file_path: str):
        """Play an audio file.

        Args:
            file_path (str): The path to the audio file to play.

        Returns:
            bool: True if playback started successfully, False otherwise.
        """
        if not os.path.isfile(file_path):
            print(f"[AudioPlayer] File does not exist: {file_path}")
            return False

        with self.lock:
            print(f"[AudioPlayer] Starting playback: {file_path}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # Wait until the audio finishes
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            print(f"[AudioPlayer] Finished playback: {file_path}")
        return True


class AudioPlayerNode(Node):
    """ROS2 Node for playing audio files on request."""

    def __init__(self):
        super().__init__("audio_player_node")
        self.get_logger().info(
            "Audio Player Node started. Waiting for audio file paths..."
        )

        self.player = AudioPlayer()

        self.subscription = self.create_subscription(
            String, "/audio_file", self.audio_file_callback, 10
        )

    def audio_file_callback(self, msg: String):
        """Callback function to handle incoming audio file path messages.

        Args:
            msg (String): The audio file path message.
        """
        file_path = msg.data.strip()
        if not file_path:
            self.get_logger().warn("Received empty file path. Ignoring.")
            return

        # Play the audio in a separate thread to avoid blocking ROS spin
        threading.Thread(
            target=self.player.play, args=(file_path,), daemon=True
        ).start()
        self.get_logger().info(f"Playback request received: {file_path}")


def main(args=None):
    """Main entry point for the Audio Player Node.

    Args:
        args : Defaults to None.
    """

    rclpy.init(args=args)
    node = AudioPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Audio Player Node.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
