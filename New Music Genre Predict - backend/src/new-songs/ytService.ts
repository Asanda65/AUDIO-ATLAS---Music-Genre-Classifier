/* eslint-disable no-console */
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { lastValueFrom } from 'rxjs';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class YtService {
  private readonly rapidApiKey =
    this.configService.get<string>('RAPID_API_KEY');
  private readonly rapidApiHost = 'spotify-scraper.p.rapidapi.com';
  private readonly requestDelay = 60000; // Delay in milliseconds (adjust as needed)

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) { }

  async getM4aLink(songName: string): Promise<string> {
    const options = {
      params: { track: songName },
      headers: {
        'X-RapidAPI-Key': this.rapidApiKey,
        'X-RapidAPI-Host': this.rapidApiHost,
      },
    };

    try {
      // Delay before making the request
      await new Promise((resolve) => setTimeout(resolve, this.requestDelay));

      const response = await lastValueFrom(
        this.httpService.get(
          'https://spotify-scraper.p.rapidapi.com/v1/track/download',
          options,
        ),
      );
      const data = response.data;
      const m4aLink = data.youtubeVideo.audio.find(
        (audio) => audio.format === 'm4a',
      )?.url;
      return m4aLink;
    } catch (error) {
      console.error(error);
      throw new Error('Failed to fetch m4a link');
    }
  }
}
