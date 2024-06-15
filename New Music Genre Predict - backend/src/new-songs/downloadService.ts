/* eslint-disable no-console */
import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import * as fetch from 'node-fetch';
import * as ffmpegPath from '@ffmpeg-installer/ffmpeg';
import * as ffmpeg from 'fluent-ffmpeg';

ffmpeg.setFfmpegPath(ffmpegPath.path);

@Injectable()
export class DownloadService {
  private readonly downloadDirectory = 'downloads';

  constructor() {
    // Create the download directory if it doesn't exist
    if (!fs.existsSync(this.downloadDirectory)) {
      fs.mkdirSync(this.downloadDirectory);
    }
  }

  async downloadAndConvertToWav(
    songName: string,
    ytLink: string,
  ): Promise<string> {
    const sanitizedSongName = sanitizeFileName(songName);
    const fileName = `${sanitizedSongName}.wav`;
    const filePath = path.join(this.downloadDirectory, fileName);
    const tempFilePath = path.join(
      this.downloadDirectory,
      `${sanitizedSongName}.m4a`,
    );

    // Download the M4A file from the YouTube link
    const downloadStream = (await fetch(ytLink)).body;
    const writeStream = fs.createWriteStream(tempFilePath);
    await new Promise((resolve, reject) => {
      downloadStream
        .pipe(writeStream)
        .on('finish', resolve)
        .on('error', reject);
    });

    // Convert the downloaded M4A file to WAV
    await new Promise<void>((resolve, reject) => {
      ffmpeg(tempFilePath)
        .audioFrequency(44100)
        .audioChannels(2)
        .audioCodec('pcm_s16le')
        .format('wav')
        .on('end', () => {
          console.log(`Downloaded and converted ${songName} to WAV`);
          fs.unlinkSync(tempFilePath); // Remove the temporary M4A file
          resolve();
        })
        .on('error', (err) => {
          console.error(`Error downloading and converting ${songName}:`, err);
          reject(err);
        })
        .save(filePath);
    });

    return filePath;
  }
}

function sanitizeFileName(fileName: string): string {
  // Replace any invalid characters with an underscore
  return fileName.replace(/[<>:"/\\|?*\x00-\x1F]/g, '_');
}
