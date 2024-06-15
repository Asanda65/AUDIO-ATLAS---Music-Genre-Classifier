import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { CreateNewSongDto } from './dto/create-new-song.dto';
import { NewSong, NewSongDocument } from './entities/new-song.entity';

@Injectable()
export class SongStorageService {
  constructor(
    @InjectModel(NewSong.name) private newSongModel: Model<NewSongDocument>,
  ) {}

  async saveSongs(createNewSongDto: CreateNewSongDto): Promise<NewSong> {
    const createdNewSong = new this.newSongModel({
      songs: createNewSongDto.songs,
      date: createNewSongDto.date,
    });

    return createdNewSong.save();
  }

  async getLatestSongs(): Promise<NewSong> {
    return this.newSongModel.findOne().sort({ date: -1 }).exec();
  }
}
