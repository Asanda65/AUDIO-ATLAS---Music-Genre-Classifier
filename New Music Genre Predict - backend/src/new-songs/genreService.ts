/* eslint-disable no-console */
import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as FormData from 'form-data';
import axios, { AxiosResponse } from 'axios';
import { CreateNewSongDto } from './dto/create-new-song.dto';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class GenreService {
  constructor(private readonly configService: ConfigService) {}
  private readonly genreApiUrl = this.configService.get<string>('GENRE_API');

  async getGenresForSongs(
    newSongDto: CreateNewSongDto,
  ): Promise<CreateNewSongDto> {
    const songsWithGenre = await Promise.all(
      newSongDto.songs.map(async (song) => {
        const genre = await this.getPredictedGenre(song.wavFilePath);
        return { ...song, genre };
      }),
    );

    return { ...newSongDto, songs: songsWithGenre };
  }

  private async getPredictedGenre(wavFilePath: string): Promise<string> {
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(wavFilePath));

    try {
      const response: AxiosResponse<{ predicted_genre: string }> =
        await axios.post(this.genreApiUrl, formData, {
          headers: {
            ...formData.getHeaders(),
          },
        });

      return response.data.predicted_genre;
    } catch (error) {
      console.error(`Error predicting genre for file ${wavFilePath}:`, error);
      throw new Error('Failed to predict genre');
    }
  }
}
